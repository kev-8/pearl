# pearl

**[pearl.dosi.io](https://pearl.dosi.io)**

Pearl is an AI research assistant created to celebrate Haiti's rich history, vibrant culture, and resilient people. Ask questions in English, French, or Haitian Creole. The name *pearl* comes from Haiti's nickname: Pearl of the Antilles. 

---

## What it does

- Searches a vector database of *Le Nouvelliste* articles spanning over a century of Haitian history
- Supplements archive results with live web search for contemporary context
- Synthesizes findings into cited, well-structured responses
- Supports multi-turn conversations with full context retention
- Responds in the language of the question (English, French, Haitian Creole)

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Dash (Python) |
| LLM | Claude Sonnet 4.5 + Haiku 4.5 (Anthropic API) |
| Embeddings | Cohere Embed v4 (Cohere API) |
| Vector DB | Pinecone |
| Web search | DuckDuckGo |
| Hosting | Railway |

## Data

The archive is built from *Le Nouvelliste* issues sourced via the Digital Libray of the Caribbean (dLOC), covering approximately 1898–1979. **[The archive is available here](https://dloc.com/UF00000081/00001)**

## Release

**v1.0.0** — February 2026
