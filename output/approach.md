# SHL Assessment Recommendation Engine - Approach Document

## 1. Problem Statement

Hiring managers and recruiters struggle to find the right SHL assessments from a catalogue of 500+ products spanning knowledge tests, cognitive aptitude, personality inventories, coding simulations, writing simulations, and pre-packaged role solutions. The current process relies on keyword filters and manual browsing, which is time-consuming and often misses relevant assessments — especially when a role requires a balanced mix of technical and behavioral evaluation.

The goal: given a natural language hiring query (e.g., *"I need a 45-minute assessment for a senior data analyst proficient in SQL, Python, and Tableau"*) or a full job description, return the top 10 most relevant SHL assessments with proper balance across assessment types.

## 2. Data Ingestion Pipeline

The foundation of the system is a clean, structured dataset scraped directly from SHL's product catalogue:

- **Web Scraper** (`core/scraper.py`): Crawls all pages of SHL's catalogue (both Individual Test Solutions and Pre-packaged Job Solutions), then visits each assessment's detail page to extract the full description, duration, job levels, languages, and test type codes. This yields **518 unique assessments** with complete metadata.
- **Embedding Index** (`core/embeddings.py`): Each assessment is converted into a rich text representation (name + description + test types + job levels) and embedded using OpenAI's `text-embedding-3-large` model (3072 dimensions). These vectors are stored in a **FAISS inner-product index** for sub-millisecond similarity search. A BM25 index is also built over the same corpus with **5x name boosting** so exact product name matches rank higher.

## 3. Recommendation Pipeline

The core pipeline (`core/graph.py`) uses **LangGraph** to orchestrate three agents with typed state flowing between them:

### Stage 1: Query Analyzer Agent

An LLM parses the input and produces:
- **15-20 search queries** — a mix of keyword queries using exact SHL product names (for BM25) and descriptive natural language queries (for FAISS semantic search)
- **Extracted skills**, **duration constraints**, and **domain label**

The prompt was designed by systematically analyzing the SHL catalogue's naming conventions. For example, SHL uses "Automata" prefixes for coding simulations, "WriteX" for writing tests, "Verify" for cognitive aptitude, "OPQ" for personality, and role-specific suffixes like "JFA", "Short Form", and "Solution". The analyzer uses these patterns to generate targeted search queries — e.g., for a "Product Manager" query, it knows to search for "Agile Software Development", "Manager 8.0 JFA", "OPQ32", and "WriteX Email Writing Managerial" alongside the semantic queries.

It also identifies implied skills: a manager role implies writing ability, a data role implies Excel proficiency, a customer-facing role implies communication assessments.

### Stage 2: Hybrid Retriever Agent

Combines two retrieval methods to cast a wide net:

- **FAISS semantic search**: Each of the 15-20 search queries is embedded and matched against the assessment index. This captures meaning-based relevance — e.g., "data analysis capabilities" matches "Tableau", "Data Warehousing", "Microsoft Excel 365".
- **BM25 keyword search**: The same queries are tokenized and matched against the name-boosted corpus. This catches exact product name matches that semantic search misses — e.g., the query "Automata Selenium" directly matches the assessment named "Automata Selenium".

**Score fusion strategy:** For each assessment, the system computes both max-score (best single query match) and sum-score (cumulative relevance across all queries) for both FAISS and BM25. These are combined as: `0.7 * max_relevance + 0.3 * breadth`, with a special BM25-only boost for strong keyword matches that FAISS ranks low, and a hit bonus for assessments found by 3+ different search queries. Per-query guaranteed slots (top 2 from each query) ensure edge-case assessments aren't lost.

This narrows 518 assessments down to **70 candidates** sent to the reranker.

### Stage 3: LLM Reranker Agent

An LLM acts as an assessment consultant, selecting exactly 10 from the 70 candidates. The selection prompt encodes principles derived from studying how SHL assessments are typically bundled for real-world roles:

1. **Named skill match** — every explicitly mentioned technology/skill gets its own test
2. **Complete coverage** — breadth across all skill areas rather than depth in one
3. **Role-fit solutions** — include JFA/Short Form packages matching the seniority level
4. **Balanced assessment types** — mix Knowledge & Skills, Simulations, Personality, and Cognitive based on what the role requires
5. **Variant awareness** — treat different versions as distinct (e.g., Professional 7.0 and 7.1 are both valid)
6. **Context-sensitive rules** — e.g., executive roles get multiple personality assessments for cultural fit, while technical roles prioritize skill tests over personality

The prompt includes representative examples spanning developers, sales, executives, administrative roles, analysts, and customer-facing positions so the LLM learns selection patterns rather than memorizing specific queries.

## 4. Technology Stack & Justification

| Component | Technology | Why |
|-----------|-----------|-----|
| Pipeline orchestration | **LangGraph** | Typed state graph with independently testable nodes; easy to extend |
| LLM | **GPT-4.1 / Gemini 3.0 Flash** | Configurable via env var; GPT for accuracy, Gemini for free-tier deployment |
| Embeddings | **OpenAI text-embedding-3-large** | 3072-dim vectors for fine-grained similarity; always uses OpenAI regardless of LLM provider |
| Vector store | **FAISS (IndexFlatIP)** | Fast cosine similarity, no external infrastructure needed |
| Keyword search | **BM25 (rank-bm25)** | Complements semantic search for exact assessment name matching |
| API | **FastAPI** | Async, auto-generated docs, Pydantic validation, matches required API spec |
| Frontend | **Streamlit** | Rapid interactive demo with sample queries and structured result cards |

## 5. Evaluation & Iteration

**Metric:** Mean Recall@10 on the labeled train set (10 queries, 65 relevant assessments).

| Iteration | What Changed | Mean Recall@10 |
|-----------|-------------|---------------|
| v1 | FAISS semantic search only | ~0.35 |
| v2 | Added BM25 hybrid retrieval with name boosting | ~0.45 |
| v3 | LLM query analyzer generating multi-query search terms | ~0.52 |
| v4 | LLM reranker with selection principles | ~0.58 |
| v5 | Score fusion tuning (Max+Sum, hit bonus, guaranteed slots) | ~0.62 |
| v6 | Refined reranker (variant awareness, role-specific patterns, balanced examples) | **~0.68** |

**Diagnostic approach:** For each train query, I separately measured (a) how many relevant assessments appear in the retriever's 70-candidate pool and (b) how many the reranker actually selects. This revealed that **95% of relevant assessments (62/65) make it into the candidate pool** — the retrieval stage has near-perfect coverage. The remaining recall gap comes from the reranker sometimes selecting related-but-not-exact assessments, making reranker prompt quality the primary lever for further improvement.

**Example improvement:** For the entry-level sales query (9 relevant assessments), early versions only selected 3 sales packages. After adding a reranker principle to "exhaust all matching role packages before adding generic tests", recall for this query jumped from 0.33 to 0.56 — the LLM now selects all 5 sales packages first, then fills remaining slots with communication assessments.

## 6. Deliverables

| Deliverable | Location | Description |
|------------|----------|-------------|
| API | `app.py` | FastAPI with `GET /health` and `POST /recommend` per spec |
| Frontend | `streamlit/streamlit_app.py` | Interactive UI with sample queries, result cards, table view |
| Pipeline | `core/graph.py` | LangGraph pipeline: QueryAnalyzer → Retriever → Reranker |
| Data scraper | `core/scraper.py` | Scrapes full SHL catalogue (518 assessments with complete descriptions) |
| Embedding builder | `core/embeddings.py` | FAISS index + BM25 index construction |
| Evaluation | `evaluate.py` | Parallel train evaluation (Recall@K) + test set prediction generation |
| Test predictions | `output/predictions.csv` | 9 test queries × 10 recommendations = 90 rows |
| Configuration | `config.py` | LLM provider toggle (GPT/Gemini), all hyperparameters |
