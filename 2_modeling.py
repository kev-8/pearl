import pandas as pd
import numpy as np
import boto3
import logging
import json
import os
import time
from pinecone import Pinecone
from ddgs import DDGS
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('botocore.credentials').setLevel(logging.WARNING)

# function to send a message to Anthropic 
def send_message_to_anthropic(messages: list[dict], 
                              model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0", 
                              region_name: str = 'us-east-1', 
                              max_tokens: int = 512,
                              tools: list[dict] = None) -> dict:
    """
    Send messages to an Anthropic model using AWS Bedrock, optionally with tools.

    Args:
        messages (list[dict]): List of message dicts, e.g., [{"role": "user", "content": "Hello"}]
        model_id (str): The model ID to use (default is Claude 3 Sonnet).
        region_name (str): The AWS region to invoke the model on.
        max_tokens (int): Maximum number of tokens to generate.
        tools (list[dict]): List of tool schemas for tool calling.

    Returns:
        dict: The full response from the model, including content and tool_calls if any.
    """
    logger.info("Sending messages to Anthropic model %s", model_id)

    accept = 'application/json'
    content_type = 'application/json'

    bedrock = boto3.client(service_name='bedrock-runtime', region_name=region_name)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": messages
    }
    if tools:
        body["tools"] = tools

    body = json.dumps(body)

    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept=accept,
            contentType=content_type
        )

        response_body = json.loads(response.get('body').read())

        logger.info("Successfully received response from Anthropic model")
        return response_body

    except ClientError as err:
        error_code = err.response["Error"]["Code"]
        message = err.response["Error"]["Message"]
        if error_code == "ThrottlingException":
            logger.warning("Throttling detected: %s. Retrying after delay.", message)
            time.sleep(5)  # Wait 5 seconds before retrying
            # Simple retry once; for more, could loop
            try:
                response = bedrock.invoke_model(
                    body=body,
                    modelId=model_id,
                    accept=accept,
                    contentType=content_type
                )
                response_body = json.loads(response.get('body').read())
                logger.info("Successfully received response after retry")
                return response_body
            except ClientError as retry_err:
                logger.error("Retry failed: %s", retry_err.response["Error"]["Message"])
                raise retry_err
        else:
            logger.error("A client error occurred: %s", message)
            raise
    except Exception as e:
        logger.error("An unexpected error occurred: %s", str(e))
        raise


# tool 1: web search engine
def web_search_engine(query: str) -> dict:
    search_results = DDGS().text(query, max_results=5)
    return search_results

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


# tool 2: pinecone vector database search
def pinecone_search(query: str) -> list[dict]:
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index('index-1')
    results = index.search(
        namespace='__default__',
        query={
            'inputs': {'text': query}, 
            'top_k': 5
        },
        rerank={
            'model': 'bge-reranker-v2-m3',
            'rank_fields': ['chunk_text'],
            'top_n': 3
        }
    )

    return results['result']['matches']

pinecone_search_tool = {
    "name": "pinecone_search",
    "description": "Searches Pinecone vector database dense index to retrieve documents relevant to the input query.",
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
    web_search_engine_tool, 
    pinecone_search_tool,
]

# map tool names to functions
functions_map = {
    "web_search_engine": web_search_engine,
    "pinecone_search": pinecone_search,
}

# agentic workflow function to call tools as needed
def run_agent(user_query: str, max_iterations: int = 5) -> str:
    """
    Completes an agentic workflow using Anthropic model and available tools.
    The agent plans, calls tools, and iterates until a final answer is reached.

    Args:
        user_query (str): The user's query.
        max_iterations (int): Maximum number of planning/execution cycles.

    Returns:
        str: The final answer from the agent.
    """
    messages = [{"role": "user", "content": user_query}]
    
    for iteration in range(max_iterations):
        logger.info(f"Iteration {iteration + 1}: Calling Anthropic model")
        
        # Call the model with tools
# agentic workflow function to call tools as needed
def run_agent(user_query: str, max_iterations: int = 5) -> str:
    """
    Completes an agentic workflow using Anthropic model and available tools.
    The agent plans, calls tools, and iterates until a final answer is reached.

    Args:
        user_query (str): The user's query.
        max_iterations (int): Maximum number of planning/execution cycles.

    Returns:
        str: The final answer from the agent.
    """
    messages = [{"role": "user", "content": user_query}]
    
    for iteration in range(max_iterations):
        logger.info(f"Iteration {iteration + 1}: Calling Anthropic model")
        
        # Call the model with tools
        response = send_message_to_anthropic(messages, tools=tools)
        
        # Extract content and tool calls
        content_blocks = response.get('content', [])
        text_content = ""
        tool_calls = []
        
        for block in content_blocks:
            if block.get('type') == 'text':
                text_content += block.get('text', '')
            elif block.get('type') == 'tool_use':
                tool_calls.append(block)
        
        # Append assistant's response to messages
        messages.append({"role": "assistant", "content": content_blocks})
        
        # If there are tool calls, execute them
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call.get('name')
                tool_input = tool_call.get('input', {})
                tool_call_id = tool_call.get('id')
                
                logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
                
                if tool_name in functions_map:
                    try:
                        result = functions_map[tool_name](**tool_input)
                        # Append tool result to messages
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_call_id": tool_call_id,
                                    "content": json.dumps(result)
                                }
                            ]
                        })
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_call_id": tool_call_id,
                                    "content": f"Error: {str(e)}"
                                }
                            ]
                        })
                else:
                    logger.error(f"Unknown tool: {tool_name}")
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_call_id": tool_call_id,
                                "content": f"Unknown tool: {tool_name}"
                            }
                        ]
                    })
        else:
            # No more tool calls, assume final answer
            logger.info("No tool calls, returning final answer")
            return text_content.strip()
        
        # Add a small delay to avoid rapid API calls
        time.sleep(1)
    
    # If max iterations reached, return the last response
    logger.warning("Max iterations reached, returning last response")
    return text_content.strip() if text_content else "Unable to complete the workflow within the iteration limit."

if __name__ == "__main__":
    query = "What happened on Air Transat flight TS663 from Port-au-Prince to Montreal?"
    answer = run_agent(query)
    print("Final Answer:", answer)


