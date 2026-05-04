# pearl

**[pearl.dosi.io](https://pearl.dosi.io)**

Pearl is an AI research assistant created to celebrate Haiti's rich history, vibrant culture, and resilient people. Ask questions in English, French, or Haitian Creole. The name *pearl* comes from Haiti's nickname: Pearl of the Antilles.

---

## What it does

- Searches a vector database of *Le Nouvelliste* articles and *Radio Haïti* broadcast transcripts spanning over a century of Haitian history
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

The archive draws from two primary sources:

- ***Le Nouvelliste*** — Haiti's oldest newspaper, sourced via the Digital Library of the Caribbean (dLOC), covering approximately 1899–1979. **[Available here](https://dloc.com/UF00000081/00001)**.
- ***Radio Haïti*** — Transcripts of Radio Haïti broadcasts from the Duke University Libraries digital repository, covering approximately 1957–2002. **[Available here](https://repository.duke.edu/dc/radiohaiti)**.

## Release

**v1.5.0** — May 2026: Added Radio Haïti broadcast transcripts (~4,400 recordings)

**v1.0.0** — February 2026: Initial release with *Le Nouvelliste* archive

---

## Kisa li ye

Pearl se yon asistan rechèch AI ki kreye pou selebre istwa rich Ayiti, kilti vibran li, ak pèp rezilyan li. Poze kesyon an kreyòl, anglè, oswa fransè. Non *pearl* soti nan ti non Ayiti: Pèl Antiy yo.

## Kisa li fè

- Fouye nan yon baz done vektè atik *Le Nouvelliste* ak transkripsyon emisyon *Radio Haïti* ki kouvri plis pase yon syèk istwa ayisyen
- Konplete rezilta achiv yo ak rechèch wèb an dirèk pou kontèks konpanporèn
- Sentetize rezilta yo nan repons ki byen site ak byen òganize
- Sipòte konvèsasyon plizyè tou ak retansyon kontèks konplè
- Reponn nan lang kesyon an (kreyòl, anglè, fransè)

## Teknoloji

| Kouch | Teknoloji |
|-------|-----------|
| Frontend | Dash (Python) |
| LLM | Claude Sonnet 4.5 + Haiku 4.5 (Anthropic API) |
| Embeddings | Cohere Embed v4 (Cohere API) |
| Baz done vektè | Pinecone |
| Rechèch wèb | DuckDuckGo |
| Ebèjman | Railway |

## Done

Achiv la soti nan de sous prensipal:

- ***Le Nouvelliste*** — pi vye jounal Ayiti, ki soti nan Bibliyotèk Nimerik Karayib la (dLOC), ki kouvri anviwon 1899–1979. **[Disponib isit la](https://dloc.com/UF00000081/00001)**.
- ***Radio Haïti*** — Transkripsyon emisyon Radio Haïti ki soti nan depo nimerik Bibliyotèk Inivèsite Duke, ki kouvri anviwon 1957–2002. **[Disponib isit la](https://repository.duke.edu/dc/radiohaiti)**.

## Piblikasyon

**v1.5.0** — Me 2026: Ajoute transkripsyon emisyon Radio Haïti (~4 400 anrejistreman)

**v1.0.0** — Fevriye 2026: Premye vèsyon ak achiv *Le Nouvelliste*

---

## Qu'est-ce que c'est

Pearl est un assistant de recherche IA créé pour célébrer la riche histoire d'Haïti, sa culture vibrante et son peuple résilient. Posez vos questions en français, en anglais, ou en créole haïtien. Le nom *pearl* vient du surnom d'Haïti : la Perle des Antilles.

## Ce qu'il fait

- Recherche dans une base de données vectorielle d'articles du *Nouvelliste* et de transcriptions d'émissions de *Radio Haïti* couvrant plus d'un siècle d'histoire haïtienne
- Complète les résultats d'archives avec une recherche web en direct pour le contexte contemporain
- Synthétise les résultats en réponses citées et bien structurées
- Prend en charge les conversations multi-tours avec rétention complète du contexte
- Répond dans la langue de la question (français, anglais, créole haïtien)

## Technologies

| Couche | Technologie |
|--------|------------|
| Frontend | Dash (Python) |
| LLM | Claude Sonnet 4.5 + Haiku 4.5 (Anthropic API) |
| Embeddings | Cohere Embed v4 (Cohere API) |
| Base vectorielle | Pinecone |
| Recherche web | DuckDuckGo |
| Hébergement | Railway |

## Données

L'archive est constituée de deux sources principales :

- ***Le Nouvelliste*** — le plus ancien journal d'Haïti, provenant de la Bibliothèque numérique des Caraïbes (dLOC), couvrant approximativement 1899–1979. **[Disponible ici](https://dloc.com/UF00000081/00001)**.
- ***Radio Haïti*** — Transcriptions d'émissions de Radio Haïti issues du dépôt numérique des bibliothèques de l'Université Duke, couvrant approximativement 1957–2002. **[Disponible ici](https://repository.duke.edu/dc/radiohaiti)**.

## Version

**v1.5.0** — Mai 2026 : Ajout des transcriptions d'émissions de Radio Haïti (~4 400 enregistrements)

**v1.0.0** — Février 2026 : Version initiale avec l'archive du *Nouvelliste*
