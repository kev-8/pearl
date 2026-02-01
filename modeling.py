from unittest import result
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
import operator
from typing import Annotated, Literal, TypedDict
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_aws import ChatBedrockConverse
from pydantic import BaseModel, Field
from preproc import generate_text_embeddings, normalize_embeddings


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('botocore.credentials').setLevel(logging.WARNING)


class AgentInput(TypedDict):
    """Simple input state for each subagent."""
    query: str


class AgentOutput(TypedDict):
    """Output from each subagent."""
    source: str
    result: str


class Classification(TypedDict):
    """A single routing decision: which agent to call with what query."""
    source: Literal["pinecone_search", "web_search"]
    query: str


class RouterState(TypedDict):
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]  
    final_answer: str

@tool
def search_pinecone(query: str) -> list[dict]:
    """Search the Pinecone vector database for relevant documents."""
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index('index-1')

    body_dict = {
        "texts": [query],
        "input_type": 'search_query',
    }

    body_json = json.dumps(body_dict, ensure_ascii=False)
    body_bytes = body_json.encode("utf-8")

    try:
        response_json = generate_text_embeddings(body_bytes)
    except ClientError as err:
        logger.error("ClientError: %s", err)
        raise

    output = response_json.get('embeddings') or response_json.get('embedding')
    embeddings = normalize_embeddings(output, expected_n=len(query))

    results = index.query(
        namespace='__default__',
        vector=embeddings[0],
        top_k=3,
        include_metadata=True
    )

    return results

@tool
def search_web(query: str) -> dict:
    """Search the web using DuckDuckGo Search API."""
    search_results = DDGS().text(query, max_results=5)
    return search_results

model = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                            max_tokens=512)

# model = ChatBedrockConverse(model="anthropic.claude-3-haiku-20240307-v1:0",
#                             max_tokens=512)

retriever_agent = create_agent(
    model=model,
    tools=[search_pinecone],
    name='retriever_agent',
    system_prompt="Use the tool to search the Pinecone vector database and retrieve relevant documents based on the user's query.",
)

web_search_agent = create_agent(
    model=model,
    tools=[search_web],
    name='web_search_agent',
    system_prompt="Use the tool to search the web and retrieve relevant content based on the user's query. Focus ONLY on credible and authoritative sources. Please cite your sources in the results.",
)

router_model = ChatBedrockConverse(model="anthropic.claude-3-haiku-20240307-v1:0",
                                  max_tokens=512)


# Define structured output schema for the classifier
class ClassificationResult(BaseModel):  
    """Result of classifying a user query into agent-specific sub-questions."""
    classifications: list[Classification] = Field(
        description="List of agents to invoke with their targeted sub-questions"
    )


def classify_query(state: RouterState) -> dict:
    """Classify query and determine which agents to invoke."""
    structured_llm = router_model.with_structured_output(ClassificationResult)  

    result = structured_llm.invoke([
        {
            "role": "system",
            "content": 
                    """Analyze this query and determine which knowledge bases to consult.
                    For each relevant source, generate a targeted sub-question optimized for that source.

                    Available sources:
                    - pinecone: historical information, archived data, past events
                    - web search engine: current events, recent news, up-to-date information

                    Return ONLY the sources that are relevant to the query. Each source should have
                    a targeted sub-question optimized for that specific knowledge domain.

                    Examples:

                    Example for "Tell me about Haiti in the 1974 World Cup":
                    - pinecone: "What historical events occurred involving Haiti in the 1974 World Cup?"
                    - web search engine: (not relevant, do not include)

                    Example for "What is the largest export of Haiti?":
                    - pinecone: "What historical data exists about Haiti's exports?"
                    - web search engine: "What are the current largest exports of Haiti?"
                    """
    },
        {"role": "user", "content": state["query"]}
    ])

    return {"classifications": result.classifications}


def route_to_agents(state: RouterState) -> list[Send]:
    """Fan out to agents based on classifications."""
    return [
        Send(c["source"], {"query": c["query"]})  
        for c in state["classifications"]
    ]


def query_pinecone(state: AgentInput) -> dict:
    """Query the Pinecone vector database agent."""""
    result = retriever_agent.invoke({
        "message": {"role": "user", "content": state["query"]}
    })
    return {"results": [{"source": "pinecone_search", "result": result["messages"][-1].content}]}


def query_web_search(state: AgentInput) -> dict:
    """Query the web search engine agent."""
    result = web_search_agent.invoke({
        "message": {"role": "user", "content": state["query"]}
    })
    return {"results": [{"source": "web_search", "result": result["messages"][-1].content}]}


def synthesize_results(state: RouterState) -> dict:
    """Combine results from all agents into a coherent answer."""
    if not state["results"]:
        return {"final_answer": "No results found from any knowledge source."}

    # Format results for synthesis
    formatted = [
        f"**From {r['source'].title()}:**\n{r['result']}"
        for r in state["results"]
    ]

    synthesis_response = router_model.invoke([
        {
            "role": "system",
            "content": f"""Synthesize these search results to answer the original question: "{state['query']}"

            - Combine information from multiple sources without redundancy
            - Highlight the most relevant and actionable information
            - Note any discrepancies between sources
            - Keep the response concise and well-organized"""
        },
        {"role": "user", "content": "\n\n".join(formatted)}
    ])

    return {"final_answer": synthesis_response.content}

# Define the overall workflow
workflow = (
    StateGraph(RouterState)
    .add_node("classify", classify_query)
    .add_node("pinecone_search", query_pinecone)
    .add_node("web_search", query_web_search)
    .add_node("synthesize", synthesize_results)
    .add_edge(START, "classify")
    .add_conditional_edges("classify", route_to_agents, ["pinecone_search", "web_search"])
    .add_edge("pinecone_search", "synthesize")
    .add_edge("web_search", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)


if __name__ == "__main__":
    query = 'Who is Mario Antonio Palacios?'
    answer = workflow.invoke({"query": query})
    print("Original query:", answer["query"])
    print("\nClassifications:")
    for c in answer["classifications"]:
        print(f"  {c['source']}: {c['query']}")
    print("\n" + "=" * 60 + "\n")
    print("Final Answer:")
    print(answer["final_answer"])


# # TODO: test agent responses for accuracy and relevance --- IGNORE ---


# # function to send a message to Anthropic 
# def send_message_to_anthropic(messages: list[dict], 
#                               model_id: str = "anthropic.claude-3-haiku-20240307-v1:0", 
#                               region_name: str = 'us-east-1', 
#                               max_tokens: int = 512,
#                               tools: list[dict] = None) -> dict:
#     """
#     Send messages to an Anthropic model using AWS Bedrock, optionally with tools.

#     Args:
#         messages (list[dict]): List of message dicts, e.g., [{"role": "user", "content": "Hello"}]
#         model_id (str): The model ID to use (default is Claude Haiku 4.5).
#         region_name (str): The AWS region to invoke the model on.
#         max_tokens (int): Maximum number of tokens to generate.
#         tools (list[dict]): List of tool schemas for tool calling.

#     Returns:
#         dict: The full response from the model, including content and tool_calls if any.
#     """
#     logger.info("Sending messages to Anthropic model %s", model_id)

#     accept = 'application/json'
#     content_type = 'application/json'

#     bedrock = boto3.client(service_name='bedrock-runtime', region_name=region_name)

#     body = {
#         "anthropic_version": "bedrock-2023-05-31",
#         "max_tokens": max_tokens,
#         "messages": messages
#     }
#     if tools:
#         body["tools"] = tools

#     request = json.dumps(body)

#     try:
#         response = bedrock.invoke_model(
#             body=request,
#             modelId=model_id,
#             accept=accept,
#             contentType=content_type
#         )

#         response_body = json.loads(response.get('body').read())

#         logger.info("Successfully received response from Anthropic model")
#         return response_body

#     except ClientError as err:
#         error_code = err.response["Error"]["Code"]
#         message = err.response["Error"]["Message"]
#         logger.error("A client error occurred: %s", message)
#         raise
#     except Exception as e:
#         logger.error("An unexpected error occurred: %s", str(e))
#         raise


# # tool 1: web search engine


# # define the tool schema
# web_search_engine_tool = {
#     "name": "web_search_engine",
#     "description": "Searches the internet and retrieves content relevant to the input query",
#     "input_schema": {
#         "type": "object",
#         "properties": {
#             "query": {
#                 "type": "string",
#                 "description": "The search query."
#             }
#         },
#         "required": ["query"]
#     }
# }


# # tool 2: pinecone vector database search
# def pinecone_search(query: str) -> list[dict]:
#     pinecone_api_key = os.getenv('PINECONE_API_KEY')
#     pc = Pinecone(api_key=pinecone_api_key)
#     index = pc.Index('index-1')

#     body_dict = {
#         "texts": [query],
#         "input_type": 'search_query',
#     }

#     body_json = json.dumps(body_dict, ensure_ascii=False)
#     body_bytes = body_json.encode("utf-8")

#     try:
#         response_json = generate_text_embeddings(body_bytes)
#     except ClientError as err:
#         logger.error("ClientError: %s", err)
#         raise

#     output = response_json.get('embeddings') or response_json.get('embedding')
#     embeddings = normalize_embeddings(output, expected_n=len(query))

#     results = index.query(
#         namespace='__default__',
#         vector=embeddings[0],
#         top_k=3,
#         include_metadata=True
#     )

#     return results


# # pinecone_search_tool = {
# #     "input_schema": {
# #         "type": "object",
# #         "properties": {
# #             "query": {
# #                 "type": "string",
# #                 "description": "The search query."
# #             }
# #         },
# #         "required": ["query"]
# #     }
# # }



# tools_list = [
#     web_search_engine_tool, 
#     pinecone_search_tool,
# ]

# # map tool names to functions
# functions_map = {
#     "web_search_engine": web_search_engine,
#     "pinecone_search": pinecone_search,
# }

# # agentic workflow function to call tools as needed
# def run_agent(user_query: str, max_iterations: int = 5) -> str:
#     """
#     Completes an agentic workflow using Anthropic model and available tools.
#     The agent plans, calls tools, and iterates until a final answer is reached.

#     Args:
#         user_query (str): The user's query.
#         max_iterations (int): Maximum number of planning/execution cycles.

#     Returns:
#         str: The final answer from the agent.
#     """
#     messages = [{"role": "user", "content": user_query}]
    
#     for iteration in range(max_iterations):
#         logger.info(f"Iteration {iteration + 1}: Calling Anthropic model")
        
#         # Call the model with tools
#         response = send_message_to_anthropic(messages, tools=tools_list)
        
#         # Extract content and tool calls
#         content_blocks = response.get('content', [])
#         text_content = ""
#         tool_calls = []
        
#         for block in content_blocks:
#             if block.get('type') == 'text':
#                 text_content += block.get('text', '')
#             elif block.get('type') == 'tool_use':
#                 tool_calls.append(block)
        
#         # Append assistant's response to messages
#         messages.append({"role": "assistant", "content": content_blocks})
        
#         # If there are tool calls, execute them
#         if tool_calls:
#             for tool_call in tool_calls:
#                 tool_name = tool_call.get('name')
#                 tool_input = tool_call.get('input', {})
#                 tool_use_id = tool_call.get('id')
                
#                 logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
                
#                 if tool_name in functions_map:
#                     try:
#                         result = functions_map[tool_name](**tool_input)
#                         # Append tool result to messages
#                         messages.append({
#                             "role": "user",
#                             "content": [
#                                 {
#                                     "type": "tool_result",
#                                     "tool_use_id": tool_use_id,
#                                     "content": json.dumps(result)
#                                 }
#                             ]
#                         })
#                     except Exception as e:
#                         logger.error(f"Error executing tool {tool_name}: {e}")
#                         messages.append({
#                             "role": "user",
#                             "content": [
#                                 {
#                                     "type": "tool_result",
#                                     "tool_use_id": tool_use_id,
#                                     "content": f"Error: {str(e)}"
#                                 }
#                             ]
#                         })
#                 else:
#                     logger.error(f"Unknown tool: {tool_name}")
#                     messages.append({
#                         "role": "user",
#                         "content": [
#                             {
#                                 "type": "tool_result",
#                                 "tool_use_id": tool_use_id,
#                                 "content": f"Unknown tool: {tool_name}"
#                             }
#                         ]
#                     })
#         else:
#             # No more tool calls, assume final answer
#             logger.info("No tool calls, returning final answer")
#             return text_content.strip()
        
#         # Add a small delay to avoid rapid API calls
#         time.sleep(1)
    
#     # If max iterations reached, return the last response
#     logger.warning("Max iterations reached, returning last response")
#     return text_content.strip() if text_content else "Unable to complete the workflow within the iteration limit."





