# load in libraries
import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import boto3
import logging
from botocore.exceptions import ClientError
from pinecone import Pinecone


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('botocore.credentials').setLevel(logging.WARNING)

# load in data 
df = pd.read_csv('/Users/kevin/Desktop/ds/other/test_df.csv')

# use text splitter to chunk articles (chunk_size=500 to be under Cohere embedding model token length)
texts_raw = df['article_text'].fillna("").astype(str).tolist()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50,
                                               is_separator_regex=False)

chunks = text_splitter.create_documents(texts_raw)

def doc_to_text_list(doc_list):
    """Convert a list of Document objects to a list of strings."""
    output = []
    for d in doc_list:
        if d is None:
            continue
        if hasattr(d, "page_content"): # check if Document-like
            output.append(str(d.page_content))
        else:
            output.append(str(d))
    return output

# get list of strings from Document objects
chunks_to_embed = doc_to_text_list(chunks) 
chunks_to_embed = [i for i in chunks_to_embed if "URL not found" not in i]

# define embedding model parameters
model_id = "cohere.embed-v4:0"
region_name = 'us-east-1'
input_type = "search_document"


def generate_text_embeddings(model_id, body, region_name):
    """
    Generate text embedding by using the Cohere Embed model.
    Args:
        model_id (str): The model ID to use.
        body (str) : The reqest body to use.
        region_name (str): The AWS region to invoke the model on
    Returns:
        dict: The response from the model.
    """

    logger.info("\n\nGenerating text embeddings with Cohere Embed %s...", model_id)

    accept = '*/*'
    content_type = 'application/json'

    bedrock = boto3.client(service_name='bedrock-runtime', region_name=region_name)

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept=accept,
        contentType=content_type
    )

    raw = response.get('body').read()
    try:
        parsed = json.loads(raw)
    except Exception:
        logger.error("Failed to parse model response. Raw (truncated): %s", raw[:2000])
        raise

    return parsed


def normalize_embeddings(embeddings, expected_n=None):
    """
    Normalize different possible shapes of embeddings into a flat list:
      - If embeddings is a dict mapping indices->vector -> convert to sorted list
    Optionally check against expected_n (number of input texts) and log a warning.
    """

    # If dict keyed by indices
    if isinstance(embeddings, dict):
        # try to convert index keys to int and sort
        try:
            items = sorted(embeddings.items(), key=lambda kv: int(kv[0]))
            flat = [v for k, v in items]
            if expected_n is not None and len(flat) != expected_n:
                logger.warning("embeddings dict length != expected_n: %d != %s", len(flat), expected_n)
            return flat
        except Exception:
            # fallback — return dict values
            flat = list(embeddings.values())
            return flat

    return flat


def main(model_id, chunks_to_embed, region_name, input_type):
    """Entrypoint for Cohere Embed example."""

    body_dict = {
        "texts": chunks_to_embed,
        "input_type": input_type,
    }

    body_json = json.dumps(body_dict, ensure_ascii=False)
    body_bytes = body_json.encode("utf-8")

    try:
        response_json = generate_text_embeddings(model_id, body_bytes, region_name)
    except ClientError as err:
        logger.error("ClientError: %s", err)
        raise

    output = response_json.get('embeddings') or response_json.get('embedding')
    embeddings = normalize_embeddings(output, expected_n=len(chunks_to_embed))

    return embeddings


if __name__ == "__main__":
    input_embeddings = main(model_id, chunks_to_embed, region_name, input_type)

# flatten if needed
if len(input_embeddings) == 1 and isinstance(input_embeddings[0], list):
    input_embeddings = input_embeddings[0]


with open('./temp_embeddings.json', 'w') as f:
    json.dump(input_embeddings, f)
print("Saved temporary embeddings to ./temp_embeddings.json")

# # load temp embeddings from file
# with open('./temp_embeddings.json', 'r') as f:
#     input_embeddings = json.load(f)


# use Pinecone vector databse to store the embeddings
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('index-1')

records = []
for i, emb in enumerate(input_embeddings):
    # build records with explicit IDs
    rec_id = f"chunk-{i}"  
    records.append({"id": rec_id, "values": emb})

try:
    # use the index upsert to store vectors
    resp = index.upsert(vectors=records)
    logger.info("Upserted %d vectors to Pinecone", len(records))
    print(f"Upserted {len(records)} vectors to Pinecone: {resp}")
except Exception as e:
    logger.error("Failed to upsert to Pinecone: %s", e)
    print("Failed to upsert to Pinecone:", e)

    