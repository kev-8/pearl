"""Stage 1 of the distillation classifier: LLM-label a chunk sample.

SCRATCH / STAGING (EVAL_LOG.md Round 18 -> distillation classifier). Labels a
few thousand real chunks with Claude Haiku via the Batch API (50% discount),
so a cheap local classifier can be trained on those labels and then run over
the full ~2.5M-chunk corpus for free. See quality_distill_train.py for stage 2.

Cost control: --dry-run prints the exact token/cost estimate and writes the
request file WITHOUT submitting anything. Always dry-run first.

Usage:
    python -m experiments.quality_distill_label --dry-run
    python -m experiments.quality_distill_label --submit
    python -m experiments.quality_distill_label --fetch <batch_id>
"""

import argparse
import json
import os
import random
import re

import pandas as pd

from preproc import chunk_dataframe
from .paths import ROOT

LE_NOUVELLISTE_CSV = ROOT / "dloc/data/output/le_nouvelliste.csv"
RADIO_HAITI_CSV = ROOT / "radiohaiti/data/output/radio_haiti.csv"

MODEL = "claude-haiku-4-5"
CHUNKS_PER_CALL = 20          # amortizes the instruction prompt across chunks
TARGET_LABELS = 5000
N_ARTICLES_PER_CORPUS = 400   # chunked, then downsampled to TARGET_LABELS
SEED = 42

REQUESTS_PATH = ROOT / "quality_distill_requests.jsonl"
SAMPLE_PATH = ROOT / "quality_distill_sample.csv"
LABELS_PATH = ROOT / "quality_distill_labels.csv"

# Batch API is 50% off list. Haiku 4.5 list: $1.00/MTok in, $5.00/MTok out.
PRICE_IN_PER_MTOK = 1.00 * 0.5
PRICE_OUT_PER_MTOK = 5.00 * 0.5

SYSTEM_PROMPT = """You are auditing text chunks extracted from two Haitian archives:
- Le Nouvelliste, a French-language newspaper (1899-1979), digitized via OCR.
- Radio Haïti, broadcast transcripts (1957-2002), transcribed via Whisper.

Chunks are ~500 characters and are often cut mid-sentence — that is NORMAL and
not a defect. Text may be in French, Haitian Creole, or a mix of both. Haitian
Creole has informal, variable orthography and legitimately uses short repeated
function words; broadcast transcripts legitimately contain songs, chants, and
rhetorical repetition. None of that makes a chunk bad.

For each chunk, assign exactly one label:

COHERENT   - Readable, meaningful text a human could understand. Counts even if
             it is cut off mid-sentence, informally spelled, mixed-language, or
             has light OCR/transcription noise. When in doubt, choose this.
GARBLED    - So corrupted it carries no recoverable meaning: OCR letter-soup,
             scrambled words, transcription loops repeating one token, or text
             in an unrelated script.
BOILERPLATE - Real words but no archival content: standalone advertisements,
             product/brand names, mastheads, page furniture, subtitle-tool
             watermarks.

Respond with ONLY a JSON array of objects, one per chunk, in the same order:
[{"i": 1, "label": "COHERENT"}, {"i": 2, "label": "GARBLED"}]
No prose, no explanation."""


def build_sample():
    """Sample real chunks from both corpora using the production chunker."""
    rows = []
    for csv_path, source in [
        (LE_NOUVELLISTE_CSV, "le_nouvelliste"),
        (RADIO_HAITI_CSV, "radio_haiti"),
    ]:
        df = pd.read_csv(csv_path)
        n = min(N_ARTICLES_PER_CORPUS, len(df))
        df_s = df.sample(n=n, random_state=SEED).reset_index(drop=True)
        chunks, _ = chunk_dataframe(df_s, shuffle=False)
        for i, text in enumerate(chunks):
            rows.append({"chunk_id": f"{source}_{i}", "source": source, "text": text})

    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows[:TARGET_LABELS]


def build_requests(sample):
    """Group chunks into batched Batch-API requests."""
    requests = []
    for start in range(0, len(sample), CHUNKS_PER_CALL):
        group = sample[start:start + CHUNKS_PER_CALL]
        numbered = "\n\n".join(
            f"--- CHUNK {i + 1} ---\n{c['text']}" for i, c in enumerate(group)
        )
        requests.append({
            "custom_id": f"grp-{start // CHUNKS_PER_CALL}",
            "chunk_ids": [c["chunk_id"] for c in group],
            "params": {
                "model": MODEL,
                "max_tokens": 2048,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": numbered}],
            },
        })
    return requests


def estimate_cost(requests):
    """Rough token/cost estimate (~4 chars per token)."""
    total_in = total_out = 0
    for r in requests:
        p = r["params"]
        chars = len(p["system"]) + len(p["messages"][0]["content"])
        total_in += chars / 4
        total_out += len(r["chunk_ids"]) * 15  # ~15 tokens per verdict object
    cost = (total_in / 1e6) * PRICE_IN_PER_MTOK + (total_out / 1e6) * PRICE_OUT_PER_MTOK
    return int(total_in), int(total_out), cost


def cmd_prepare(dry_run: bool):
    sample = build_sample()
    requests = build_requests(sample)
    tin, tout, cost = estimate_cost(requests)

    with open(REQUESTS_PATH, "w") as f:
        for r in requests:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(sample).to_csv(SAMPLE_PATH, index=False)

    print(f"Chunks to label : {len(sample)}")
    print(f"Batched calls   : {len(requests)} ({CHUNKS_PER_CALL} chunks/call)")
    print(f"Est. input tok  : {tin:,}")
    print(f"Est. output tok : {tout:,}")
    print(f"Est. cost       : ${cost:.2f}  (Batch API, 50% off {MODEL})")
    print(f"Wrote {REQUESTS_PATH} and {SAMPLE_PATH}")
    if dry_run:
        print("\nDRY RUN — nothing submitted. Re-run with --submit to send.")


def cmd_submit():
    import anthropic
    client = anthropic.Anthropic()

    requests = [json.loads(l) for l in open(REQUESTS_PATH)]
    _, _, cost = estimate_cost(requests)
    print(f"Submitting {len(requests)} batched calls (est. ${cost:.2f})...")

    batch = client.messages.batches.create(
        requests=[
            {"custom_id": r["custom_id"], "params": r["params"]} for r in requests
        ]
    )
    print(f"Batch submitted: {batch.id}")
    print(f"Status: {batch.processing_status}")
    print(f"\nWhen it finishes:  python -m experiments.quality_distill_label --fetch {batch.id}")


def _parse_labels(text):
    """Extract the JSON array of verdicts from a model response."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def cmd_fetch(batch_id: str):
    import anthropic
    client = anthropic.Anthropic()

    batch = client.messages.batches.retrieve(batch_id)
    print(f"Status: {batch.processing_status}")
    if batch.processing_status != "ended":
        print("Not finished yet — try again later.")
        return

    by_custom_id = {
        json.loads(l)["custom_id"]: json.loads(l) for l in open(REQUESTS_PATH)
    }
    sample = pd.read_csv(SAMPLE_PATH).set_index("chunk_id")

    out, failed = [], 0
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            failed += 1
            continue
        req = by_custom_id.get(result.custom_id)
        if not req:
            continue
        text = "".join(
            b.text for b in result.result.message.content if b.type == "text"
        )
        verdicts = _parse_labels(text)
        if not verdicts:
            failed += 1
            continue
        for v in verdicts:
            idx = v.get("i", 0) - 1
            if 0 <= idx < len(req["chunk_ids"]):
                cid = req["chunk_ids"][idx]
                if cid in sample.index:
                    out.append({
                        "chunk_id": cid,
                        "source": sample.loc[cid, "source"],
                        "label": str(v.get("label", "")).strip().upper(),
                        "text": sample.loc[cid, "text"],
                    })

    df = pd.DataFrame(out).drop_duplicates(subset="chunk_id")
    df = df[df["label"].isin(["COHERENT", "GARBLED", "BOILERPLATE"])]
    df.to_csv(LABELS_PATH, index=False)

    print(f"Labeled {len(df)} chunks ({failed} failed groups)")
    print(df["label"].value_counts())
    print(f"Wrote {LABELS_PATH}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="estimate cost, submit nothing")
    g.add_argument("--submit", action="store_true", help="submit the batch job")
    g.add_argument("--fetch", metavar="BATCH_ID", help="retrieve finished labels")
    args = ap.parse_args()

    if args.dry_run:
        cmd_prepare(dry_run=True)
    elif args.submit:
        if not os.path.exists(REQUESTS_PATH):
            cmd_prepare(dry_run=False)
        cmd_submit()
    else:
        cmd_fetch(args.fetch)


if __name__ == "__main__":
    main()
