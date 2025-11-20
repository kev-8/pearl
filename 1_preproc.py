# load in libraries
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import boto3
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# load in data 
df = pd.read_csv('/Users/kevin/Desktop/ds/other/test_df.csv')


# use text splitter to chunk articles (chunk_size=500 to be under Cohere embedding model token length)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50,
                                               separators=["\n\n", "\n", " ", ""])

texts = text_splitter.create_documents(df['article_text'])

chunks = text_splitter.split_documents(documents=texts)
print(f'Split into {len(chunks)} chunks')

# convert chunks into json
chunks = json.dumps([{'page_content': doc.page_content} for doc in chunks], indent=4)

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

    logger.info("\nGenerating text embeddings with the Cohere Embed model %s", model_id)

    accept = '*/*'
    content_type = 'application/json'

    bedrock = boto3.client(service_name='bedrock-runtime', region_name=region_name)

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept=accept,
        contentType=content_type
    )

    logger.info("\nSuccessfully generated embeddings with Cohere model %s", model_id)

    return response


def main(model_id, chunks, region_name, input_type):
    """
    Entrypoint for Cohere Embed example.
    """

    # logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        body = json.dumps({
            "texts": chunks,
            "input_type": input_type,
        })
        
        response = generate_text_embeddings(model_id, body, region_name)

        response_body = json.loads(response.get('body').read())

        print(f"ID: {response_body.get('id')}")
        print(f"Response type: {response_body.get('response_type')}")

        print("Embeddings")
        embeddings = response_body.get('embeddings')
        for i, embedding_type in enumerate(embeddings):
            print(f"\t{embedding_type} Embeddings:")
            print(f"\t{embeddings[embedding_type]}")

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " +
              format(message))
    else:
        print(
            f"Finished generating text embeddings with Cohere model {model_id}.")


if __name__ == "__main__":
    main(model_id, chunks, region_name, input_type)


# use pinecone to store the embeddings
