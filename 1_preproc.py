# load in libraries
import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import boto3
import logging
from botocore.exceptions import ClientError
from pinecone import Pinecone
import uuid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('botocore.credentials').setLevel(logging.WARNING)

# load in data 
# df = pd.read_csv('/Users/kevin/Desktop/ds/other/test_df.csv')

# # convert column to a list of clean strings
# texts_raw = df['article_text'].fillna("").astype(str).tolist()

# # use text splitter to chunk articles (chunk_size=500 to be under Cohere embedding model token length)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
#                                                chunk_overlap=50,
#                                                is_separator_regex=False)

# chunks = text_splitter.create_documents(texts_raw)

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


# chunks_to_embed = doc_to_text_list(chunks)  # list[str]

# # define embedding model parameters
# model_id = "cohere.embed-v4:0"
# region_name = 'us-east-1'
# input_type = "search_document"


# def generate_text_embeddings(model_id, body, region_name):
#     """
#     Generate text embedding by using the Cohere Embed model.
#     Args:
#         model_id (str): The model ID to use.
#         body (str) : The reqest body to use.
#         region_name (str): The AWS region to invoke the model on
#     Returns:
#         dict: The response from the model.
#     """

#     logger.info("\n\nGenerating text embeddings with Cohere Embed %s", model_id)

#     accept = '*/*'
#     content_type = 'application/json'

#     bedrock = boto3.client(service_name='bedrock-runtime', region_name=region_name)

#     response = bedrock.invoke_model(
#         body=body,
#         modelId=model_id,
#         accept=accept,
#         contentType=content_type
#     )

#     logger.info("\n\nSuccessfully generated embeddings with model: %s", model_id)

#     return response


# def main(model_id, chunks_to_embed, region_name, input_type):
#     """Entrypoint for Cohere Embed example."""

#     try:
#         body = json.dumps({
#             "texts": chunks_to_embed,
#             "input_type": input_type,
#         })
        
#         response = generate_text_embeddings(model_id, body, region_name)
#         response_body = json.loads(response.get('body').read())
#         embeddings = response_body.get('embeddings')

#     except ClientError as err:
#         message = err.response["Error"]["Message"]
#         logger.error("A client error occurred: %s", message)
#         print("A client error occured: " +
#               format(message))
        
#     else:
#         return embeddings



# if __name__ == "__main__":
#     input_embeddings = main(model_id, chunks_to_embed, region_name, input_type)

# save temp embeddings to file
# with open('./temp_embeddings.json', 'w') as f:
#     json.dump(input_embeddings, f)
# print("Saved temporary embeddings to ./temp_embeddings.json")

# load temp embeddings from file
with open('./temp_embeddings.json', 'r') as f:
    input_embeddings = json.load(f)
print("Loaded temporary embeddings from ./temp_embeddings.json")

# use pinecone to store the embeddings
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index('pearl-index')

if input_embeddings:
    # build Pinecone records with explicit IDs and metadata
    records = []
    for i, emb in enumerate(input_embeddings):
        rec_id = f"chunk-{i}"
        # Pinecone expects a dict with id, values, and optional metadata
        records.append({"id": rec_id, "values": emb})

    try:
        # Use the index upsert to store vectors
        resp = index.upsert(vectors=records)
        logger.info("Upserted %d vectors to Pinecone", len(records))
        print(f"Upserted {len(records)} vectors to Pinecone: {resp}")
    except Exception as e:
        logger.error("Failed to upsert to Pinecone: %s", e)
        print("Failed to upsert to Pinecone:", e)