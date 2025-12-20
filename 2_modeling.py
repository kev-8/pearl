# load in libraries
import pandas as pd
import numpy as np
import boto3
import logging
import json
from pinecone import Pinecone
from ddgs import DDGS
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('botocore.credentials').setLevel(logging.WARNING)


# tool 1: web search engine
def web_search_engine(query: str) -> dict:
    search_results = DDGS().text(query, max_results=5)
    return search_results

# map tool names to functions
functions_map = {
    "web_search_engine": web_search_engine,
}

# define the tool schema
web_search_engine_tool = {
    "name": "web_search_engine",
    "description": "Searches the internet and retrieves content relevant to the input query",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query."
            }
        },
        "required": ["query"]
    }
}

tools = [
  web_search_engine_tool
]

# function to send a message to Anthropic 
def send_message_to_anthropic(message: str, model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0", region_name: str = 'us-east-1', max_tokens: int = 1000) -> str:
    """
    Send a message to an Anthropic model using AWS Bedrock.

    Args:
        message (str): The message to send to the model.
        model_id (str): The model ID to use (default is Claude 3 Sonnet).
        region_name (str): The AWS region to invoke the model on.
        max_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: The response from the model.
    """
    logger.info("Sending message to Anthropic model %s", model_id)

    accept = 'application/json'
    content_type = 'application/json'

    bedrock = boto3.client(service_name='bedrock-runtime', region_name=region_name)

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ]
    })

    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept=accept,
            contentType=content_type
        )

        response_body = json.loads(response.get('body').read())
        output = response_body.get('content', [{}])[0].get('text', '')

        logger.info("Successfully received response from Anthropic model")
        return output

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        raise
    except Exception as e:
        logger.error("An unexpected error occurred: %s", str(e))
        raise






