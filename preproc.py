import os
import sys
import argparse
import time
import json
import logging

import cohere
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

METADATA_COLS = ["SQLDATE", "source_url", "issue_id", "top_entity_names", "top_entity_labels"]
EMBED_BATCH_SIZE = 96       # Bedrock Cohere Embed v4 limit
UPSERT_BATCH_SIZE = 100     # Pinecone recommended batch size
MAX_META_BYTES = 40_000     # Pinecone 40KB metadata limit per vector
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0           # seconds, exponential backoff base
THROTTLE_RETRY_DELAY = 60.0      # seconds, delay for throttling/rate-limit errors
THROTTLE_MAX_RETRIES = 10        # more attempts for throttling since it may clear


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def doc_to_text_list(doc_list):
    """Convert a list of Document objects to a list of strings."""
    output = []
    for d in doc_list:
        if d is None:
            continue
        if hasattr(d, "page_content"):
            output.append(str(d.page_content))
        else:
            output.append(str(d))
    return output


def generate_text_embeddings(body, model_id="embed-v4.0", region_name='us-east-1'):
    """
    Generate text embeddings using the Cohere API.
    Args:
        body (bytes): JSON-encoded request body with 'texts' and 'input_type' keys.
        model_id (str): The Cohere model ID to use.
        region_name (str): Unused; kept for interface compatibility.
    Returns:
        dict: {'embeddings': list_of_vectors}
    """
    body_dict = json.loads(body)
    texts = body_dict.get('texts', [])
    input_type = body_dict.get('input_type', 'search_document')

    co = cohere.ClientV2(api_key=os.getenv('COHERE_API_KEY'))
    response = co.embed(
        texts=texts,
        model=model_id,
        input_type=input_type,
        embedding_types=['float'],
    )
    return {'embeddings': response.embeddings.float_}


def normalize_embeddings(embeddings, expected_n=None):
    """
    Normalize different possible shapes of embeddings into a flat list:
      - If embeddings is a dict mapping indices->vector -> convert to sorted list
      - If already a list, return as-is
    Optionally check against expected_n (number of input texts) and log a warning.
    """
    if isinstance(embeddings, dict):
        try:
            items = sorted(embeddings.items(), key=lambda kv: int(kv[0]))
            flat = [v for k, v in items]
            if expected_n is not None and len(flat) != expected_n:
                logger.warning("embeddings dict length != expected_n: %d != %s", len(flat), expected_n)
            return flat
        except Exception:
            return list(embeddings.values())

    if isinstance(embeddings, list):
        if expected_n is not None and len(embeddings) != expected_n:
            logger.warning("embeddings list length != expected_n: %d != %s", len(embeddings), expected_n)
        return embeddings

    raise ValueError(f"Unexpected embeddings type: {type(embeddings)}")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_dataframe(df, shuffle=False, seed=42):
    """Split article_text into chunks and build chunk-to-row index mapping.

    If shuffle=True, rows are randomly permuted (with a fixed seed) before
    chunking so that the resulting chunks span all time periods evenly.
    """
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        logger.info("Shuffled DataFrame rows (seed=%d)", seed)

    texts_raw = df['article_text'].fillna("").astype(str).tolist()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, is_separator_regex=False
    )

    chunks = text_splitter.create_documents(texts_raw)

    chunk_row_indices = []
    for row_idx, text in enumerate(texts_raw):
        row_chunks = text_splitter.split_text(text)
        chunk_row_indices.extend([row_idx] * len(row_chunks))

    chunks_text = doc_to_text_list(chunks)
    chunks_text = [t for t in chunks_text if "URL not found" not in t]

    logger.info("Chunking complete: %d chunks from %d articles", len(chunks_text), len(df))
    return chunks_text, chunk_row_indices


# ---------------------------------------------------------------------------
# Batch embedding with checkpoint/resume
# ---------------------------------------------------------------------------

def _is_throttling_error(exc):
    """Check if an exception is a throttling / rate-limit error."""
    exc_str = str(exc).lower()
    return any(k in exc_str for k in ["throttl", "too many", "rate limit", "429"])


def _embed_batch_with_retry(texts):
    """Embed a single batch of texts with exponential-backoff retry.

    Uses longer delays and more attempts for throttling errors.
    """
    body_dict = {"texts": texts, "input_type": "search_document"}
    body_bytes = json.dumps(body_dict, ensure_ascii=False).encode("utf-8")

    max_attempts = MAX_RETRIES
    for attempt in range(1, THROTTLE_MAX_RETRIES + 1):
        try:
            response_json = generate_text_embeddings(body_bytes)
            raw_emb = response_json.get('embeddings') or response_json.get('embedding')
            return normalize_embeddings(raw_emb, expected_n=len(texts))
        except Exception as e:
            throttled = _is_throttling_error(e)
            max_attempts = THROTTLE_MAX_RETRIES if throttled else MAX_RETRIES

            if attempt >= max_attempts:
                logger.error("Embedding failed after %d retries: %s", attempt, e)
                raise

            if throttled:
                delay = THROTTLE_RETRY_DELAY * attempt  # 60s, 120s, 180s, ...
                logger.warning("Throttled (attempt %d/%d), waiting %.0fs...",
                               attempt, max_attempts, delay)
            else:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning("Embed attempt %d failed (%s), retrying in %.1fs...",
                               attempt, e, delay)
            time.sleep(delay)


def embed_all_chunks(chunks_text, batch_size, checkpoint_path, resume):
    """Embed all chunks in batches, writing results to a JSONL checkpoint file."""
    total_batches = (len(chunks_text) + batch_size - 1) // batch_size

    completed_batches = 0
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            completed_batches = sum(1 for _ in f)
        logger.info("Resuming from batch %d / %d", completed_batches, total_batches)

    mode = 'a' if resume and os.path.exists(checkpoint_path) else 'w'
    with open(checkpoint_path, mode) as cp_file:
        for batch_idx in range(completed_batches, total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(chunks_text))
            batch_texts = chunks_text[start:end]

            embeddings = _embed_batch_with_retry(batch_texts)

            # flatten if needed (single-element wrapper list)
            if len(embeddings) == 1 and isinstance(embeddings[0], list) and len(batch_texts) > 1:
                embeddings = embeddings[0]

            cp_file.write(json.dumps(embeddings) + '\n')
            cp_file.flush()

            if (batch_idx + 1) % 100 == 0 or batch_idx == total_batches - 1:
                logger.info("Embedded batch %d / %d  (%d chunks so far)",
                            batch_idx + 1, total_batches, end)

            time.sleep(0.1)  # rate-limit courtesy delay

    logger.info("Embedding complete: %d batches written to %s", total_batches, checkpoint_path)


# ---------------------------------------------------------------------------
# Batch Pinecone upsert
# ---------------------------------------------------------------------------

def _upsert_batch_with_retry(index, batch):
    """Upsert a single batch of vectors to Pinecone with retry."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            index.upsert(vectors=batch)
            return
        except Exception as e:
            if attempt == MAX_RETRIES:
                logger.error("Upsert failed after %d retries: %s", MAX_RETRIES, e)
                raise
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning("Upsert attempt %d failed (%s), retrying in %.1fs...", attempt, e, delay)
            time.sleep(delay)


def _build_record(emb, chunk_text, chunk_row_idx, df, global_idx):
    """Build a single Pinecone record with metadata."""
    meta = {"text": chunk_text}

    if chunk_row_idx is not None:
        row = df.iloc[chunk_row_idx]
        for col in METADATA_COLS:
            if col in df.columns and pd.notna(row.get(col)):
                meta[col.lower()] = str(row[col])

    # Truncate text metadata if it exceeds Pinecone's 40KB limit
    meta_bytes = len(json.dumps(meta).encode('utf-8'))
    if meta_bytes > MAX_META_BYTES:
        overflow = meta_bytes - MAX_META_BYTES + 100
        meta["text"] = meta["text"][:-overflow] + "..."
        logger.warning("Truncated metadata for chunk-%d (was %d bytes)", global_idx, meta_bytes)

    return {"id": f"chunk-{global_idx}", "values": emb, "metadata": meta}


def upsert_from_checkpoint(checkpoint_path, chunks_text, chunk_row_indices, df,
                           upsert_progress_path, resume, max_vectors=None, id_offset=0):
    """Stream embeddings from checkpoint JSONL and upsert in batches.

    Reads one JSONL line at a time to avoid loading the full file into memory.
    """
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index('index-1')

    # Resume: read how many vectors were already upserted
    vectors_upserted = 0
    if resume and os.path.exists(upsert_progress_path):
        with open(upsert_progress_path, 'r') as f:
            try:
                vectors_upserted = int(f.read().strip())
            except ValueError:
                vectors_upserted = 0
        logger.info("Resuming upsert from vector %d", vectors_upserted)

    batch = []
    vector_idx = 0       # position within the checkpoint stream
    upserted_count = 0   # vectors upserted this run
    batch_count = 0

    with open(checkpoint_path, 'r') as f:
        for line in f:
            embeddings = json.loads(line)
            for emb in embeddings:
                # Skip vectors already upserted
                if vector_idx < vectors_upserted:
                    vector_idx += 1
                    continue

                # Stop if we've hit the max
                if max_vectors and upserted_count >= max_vectors:
                    break

                global_id = id_offset + vector_idx
                chunk_text = chunks_text[vector_idx] if vector_idx < len(chunks_text) else ""
                row_idx = chunk_row_indices[vector_idx] if vector_idx < len(chunk_row_indices) else None

                batch.append(_build_record(emb, chunk_text, row_idx, df, global_id))
                vector_idx += 1
                upserted_count += 1

                if len(batch) >= UPSERT_BATCH_SIZE:
                    _upsert_batch_with_retry(index, batch)
                    batch = []
                    batch_count += 1

                    with open(upsert_progress_path, 'w') as pf:
                        pf.write(str(vectors_upserted + upserted_count))

                    if batch_count % 50 == 0:
                        logger.info("Upserted %d vectors so far (%d batches)",
                                    upserted_count, batch_count)

            if max_vectors and upserted_count >= max_vectors:
                break

    # Flush remaining
    if batch:
        _upsert_batch_with_retry(index, batch)
        batch_count += 1
        with open(upsert_progress_path, 'w') as pf:
            pf.write(str(vectors_upserted + upserted_count))

    logger.info("Upsert complete: %d vectors to Pinecone (%d batches)", upserted_count, batch_count)
    return index


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embed and upsert Le Nouvelliste chunks")
    parser.add_argument("--csv-path", default="dloc/data/output/le_nouvelliste.csv",
                        help="Path to the input CSV")
    parser.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE,
                        help=f"Embedding batch size (max {EMBED_BATCH_SIZE})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint file")
    parser.add_argument("--checkpoint", default="embeddings_checkpoint.jsonl",
                        help="Path to the JSONL checkpoint file")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle DataFrame rows before chunking for temporal diversity")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffle (default: 42)")
    parser.add_argument("--max-vectors", type=int, default=None,
                        help="Max vectors to upsert (cap for Pinecone Standard WU budget)")
    parser.add_argument("--skip-vectors", type=int, default=0,
                        help="Skip the first N chunks (already processed in a prior run)")
    parser.add_argument("--upsert-only", action="store_true",
                        help="Skip embedding, stream upsert from existing checkpoint file")
    args = parser.parse_args()

    if args.batch_size > EMBED_BATCH_SIZE:
        logger.warning("Batch size %d exceeds Bedrock limit of %d, clamping",
                        args.batch_size, EMBED_BATCH_SIZE)
        args.batch_size = EMBED_BATCH_SIZE

    # 1. Load data
    logger.info("Loading CSV: %s", args.csv_path)
    df = pd.read_csv(args.csv_path)
    logger.info("Loaded %d articles", len(df))

    # 2. Chunk (optionally shuffle for temporal diversity)
    chunks_text, chunk_row_indices = chunk_dataframe(df, shuffle=args.shuffle, seed=args.seed)

    # 3. Skip already-processed chunks from prior runs
    if args.skip_vectors > 0:
        logger.info("Skipping first %d chunks (--skip-vectors)", args.skip_vectors)
        chunks_text = chunks_text[args.skip_vectors:]
        chunk_row_indices = chunk_row_indices[args.skip_vectors:]

    # 4. Cap chunks to max-vectors to avoid embedding more than we'll upsert
    if args.max_vectors and args.max_vectors < len(chunks_text):
        logger.info("Capping to %d / %d remaining chunks (--max-vectors)", args.max_vectors, len(chunks_text))
        chunks_text = chunks_text[:args.max_vectors]
        chunk_row_indices = chunk_row_indices[:args.max_vectors]

    # 5. Embed in batches (skip if --upsert-only)
    if not args.upsert_only:
        embed_all_chunks(chunks_text, args.batch_size, args.checkpoint, args.resume)

    # 6. Stream upsert from checkpoint file
    upsert_progress_path = args.checkpoint.replace('.jsonl', '_upsert_progress.txt')
    index = upsert_from_checkpoint(args.checkpoint, chunks_text, chunk_row_indices, df,
                                   upsert_progress_path, args.resume, args.max_vectors,
                                   id_offset=args.skip_vectors)

    # 6. Report
    stats = index.describe_index_stats()
    logger.info("Pinecone index stats: %s", stats)


if __name__ == "__main__":
    main()
