import pandas as pd
import numpy as np
import logging
import json
import os
from pinecone import Pinecone
from ddgs import DDGS
from botocore.exceptions import ClientError
import operator
from typing import Annotated, Literal, TypedDict
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_aws import ChatBedrockConverse
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from preproc import generate_text_embeddings, normalize_embeddings


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('botocore.credentials').setLevel(logging.WARNING)

_pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
_index = _pc.Index('index-1')


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
def search_pinecone(query: str) -> str:
    """Search the Pinecone vector database for relevant documents."""
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
    embeddings = normalize_embeddings(output, expected_n=1)

    results = _index.query(
        namespace='__default__',
        vector=embeddings[0],
        top_k=6,
        include_metadata=True
    )

    lines = []
    for i, match in enumerate(results.get('matches', []), start=1):
        meta = match.get('metadata', {})
        date = meta.get('date', 'unknown date')
        source = meta.get('source', '')
        entities = meta.get('entities', '')
        text = meta.get('text', '').strip()
        header = f"[{i}] Date: {date}"
        if source:
            header += f" | Source: {source}"
        lines.append(header)
        if entities:
            lines.append(f"    Entities: {entities}")
        if text:
            lines.append(f'    "{text}"')
        lines.append('')

    return '\n'.join(lines) if lines else 'No results found.'

@tool
def search_web(query: str) -> dict:
    """Search the web using DuckDuckGo Search API."""
    search_results = DDGS().text(query, max_results=5)
    return search_results

# using haiku for testng 
model = ChatBedrockConverse(model="anthropic.claude-3-haiku-20240307-v1:0")

retriever_agent = create_agent(
    model=model,
    tools=[search_pinecone],
    name='retriever_agent',
    system_prompt=(
        "You are a specialist in the Le Nouvelliste archive — a Haitian newspaper published continuously since 1898, "
        "written primarily in French and Haitian Creole. Your job is to search the archive and extract relevant "
        "historical facts. Always cite the article date and source URL from the metadata. Organize your findings "
        "clearly, noting the time period and context."
    ),
)

web_search_agent = create_agent(
    model=model,
    tools=[search_web],
    name='web_search_agent',
    system_prompt=(
        "You are searching for supplementary and current information about Haiti to complement a historical archive. "
        "Prioritize credible, authoritative sources (academic institutions, reputable journalism, government and NGO reports). "
        "Always cite the source URL for every claim. Organize results clearly."
    ),
)

router_model = ChatBedrockConverse(model="anthropic.claude-3-haiku-20240307-v1:0")

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
            "content": (
                "You route user questions about Haiti to the right knowledge sources:\n"
                "- `pinecone_search`: historical information from Le Nouvelliste, Haiti's oldest newspaper "
                "(articles from ~1900 to present). Use for Haitian history, culture, politics, economics, "
                "social events, named figures, and places documented in the archive.\n"
                "- `web_search`: current events, recent context, or topics unlikely to appear in historical "
                "newspaper archives (e.g., post-2020 events, technical/scientific topics, comparative context).\n"
                "For each relevant source, write a targeted sub-question optimized for that source's strengths. "
                "Only return sources that are genuinely relevant."
            ),
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
    """Query the Pinecone vector database agent."""
    result = retriever_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"results": [{"source": "pinecone_search", "result": result["messages"][-1].content}]}


def query_web_search(state: AgentInput) -> dict:
    """Query the web search engine agent."""
    result = web_search_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"results": [{"source": "web_search", "result": result["messages"][-1].content}]}


def has_conflicting_results(results: list[AgentOutput]) -> bool:
    """Check if results from different sources contain conflicting information."""
    if len(results) <= 1:
        return False
    
    # Multiple sources means potential conflicts requiring synthesis
    sources = [r['source'] for r in results]
    return len(set(sources)) > 1


def synthesize_results(state: RouterState) -> dict:
    """Combine results from all agents into a coherent answer."""
    if not state["results"]:
        return {"final_answer": "No results found from any knowledge source."}

    # If single source or no conflicts, return directly without synthesis
    if not has_conflicting_results(state["results"]):
        return {"final_answer": state["results"][0]["result"]}

    # Format results for synthesis only when needed
    formatted = [
        f"**From {r['source'].title()}:**\n{r['result']}"
        for r in state["results"]
    ]

    synthesis_response = router_model.invoke([
        {
            "role": "system",
            "content": (
                "You are synthesizing research about Haiti from multiple sources to answer the user's question. "
                "Write a clear, well-structured response in markdown. Use the following format:\n"
                "- A brief direct answer to the question (1-2 sentences)\n"
                "- Organized sections with headers if multiple aspects are covered\n"
                "- Bullet points for lists of facts, events, or entities\n"
                "- Source citations inline (e.g., *Le Nouvelliste, 1947-03-12* or [Source](url))\n"
                "- A \"Historical Context\" section when archival data provides meaningful background\n"
                "Avoid redundancy. Note discrepancies between sources if relevant. "
                "Keep the response focused and readable."
            ),
        },
        {"role": "user", "content": f'Question: {state["query"]}\n\n' + "\n\n".join(formatted)}
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


# state tracking so the agent can engage in multi-turn conversations
@tool
def search_knowledge_sources(query: str) -> str:
    """Search both Pinecone and web sources and return combined results to answer the query."""
    answer = workflow.invoke({"query": query})
    return answer["final_answer"]

conversational_agent = create_agent(
    model=model,
    tools=[search_knowledge_sources],
    system_prompt=(
        "You are Pearl, an AI research assistant specialized in Haitian history and culture, powered by the "
        "Le Nouvelliste archive — Haiti's oldest newspaper, published since 1898 — and live web search. "
        "You can answer questions in English, French, and Haitian Creole. "
        "Use the `search_knowledge_sources` tool for every substantive question. "
        "Synthesize historical archive results with contemporary context when relevant. "
        "Be precise about dates and sources. When the user's question is ambiguous, ask for clarification before searching."
    ),
    checkpointer=InMemorySaver()
)

# if __name__ == "__main__":
#     query = 'Tell me more about the other significant industries.'

#     config = {"configurable": {"thread_id": "thread-1"}}
#     answer = conversational_agent.invoke(
#         {"messages": [{"role": "user", "content": query}]},
#         config=config)

#     print("\nFinal Answer:\n", answer["messages"][-1].content)


