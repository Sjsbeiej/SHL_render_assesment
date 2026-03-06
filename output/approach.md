# SHL Assessment Recommendation Engine - Approach Document

## 1. Problem Statement

Hiring managers need a fast way to find the right SHL assessments for a given role. The current catalogue has 500+ assessments across skill tests, cognitive aptitude, personality, simulations, and pre-packaged solutions. Manually browsing through filters is slow and error-prone. The goal is to build a system that takes a natural language hiring query or job description and returns the 10 most relevant assessments.

## 2. Solution Approach

I designed a **three-stage pipeline** that mirrors how a human assessment consultant would work:

**Stage 1 - Understand the Query:** An LLM reads the input and identifies what the role needs — explicit skills (Python, SQL), role level (entry/mid/senior), implied requirements (a data role needs Excel, a manager needs writing skills), and any constraints (duration limits). It then generates multiple search queries to cast a wide net across the assessment catalogue.

**Stage 2 - Find Candidates:** A hybrid search combines two complementary methods:
- **Semantic search** (FAISS vectors) to find assessments whose descriptions match the meaning of the query
- **Keyword search** (BM25) to find assessments whose names exactly match mentioned skills

This hybrid approach is critical because semantic search alone misses exact product names (e.g., "Automata Selenium"), while keyword search alone misses implied requirements. The two are fused using a weighted scoring strategy that rewards assessments found by multiple queries. This narrows 500+ assessments to ~70 candidates.

**Stage 3 - Select the Best 10:** An LLM acts as an assessment consultant, selecting exactly 10 from the 70 candidates. The selection follows principles I developed by studying how SHL assessments are typically bundled for different roles:
- Every named skill gets its own test
- Coverage across all skill areas (breadth over depth)
- Include role-fit packages matching the seniority level
- Balance hard skills (Knowledge & Skills, Simulations) with soft skills (Personality, Cognitive) based on what the role actually requires

## 3. Key Design Decisions

**Why hybrid retrieval?** Pure semantic search gave ~35% recall. Adding BM25 keyword matching with assessment name boosting pushed it to ~50%. The combination catches both "what the role means" and "what specific products exist."

**Why a separate reranker?** The retriever optimizes for broad coverage (find anything potentially relevant). The reranker optimizes for selection quality (pick the best balanced set). Separating these concerns gave better results than trying to do both in one step.

**Why LangGraph?** The pipeline has clear stages with typed state flowing between them. LangGraph makes each stage independently testable and the overall flow easy to extend (e.g., adding a URL-fetching stage for JD links).

**Why configurable LLM provider?** The system supports both GPT and Gemini (configurable via environment variable). Embeddings always use OpenAI for consistency with the pre-built FAISS index, while LLM calls can use either provider. This allows using Gemini's free tier for cost-effective deployment.

## 4. Evaluation & Iteration

I used **Mean Recall@10** on the labeled train set (10 queries, 65 relevant assessments) to iterate.

| Iteration | What Changed | Recall@10 |
|-----------|-------------|-----------|
| v1 | Semantic search only | ~0.35 |
| v2 | Added BM25 hybrid search | ~0.45 |
| v3 | Added LLM query analyzer for multi-query search | ~0.52 |
| v4 | Added LLM reranker with selection principles | ~0.58 |
| v5 | Tuned score fusion, added name boosting, hit bonus | ~0.62 |
| v6 | Refined reranker principles (variant awareness, role-specific patterns) | **~0.68** |

**Key insight:** I found that 95% of relevant assessments (62/65) make it into the retriever's candidate pool. The remaining recall gap is primarily a reranker selection problem — the LLM sometimes picks related but not exactly matching assessments. This means further improvement should focus on reranker prompt quality rather than retrieval tuning.

## 5. Deliverables

| Deliverable | Location |
|------------|----------|
| API (FastAPI) | `app.py` — `/health` and `/recommend` endpoints |
| Frontend (Streamlit) | `streamlit/streamlit_app.py` |
| Pipeline code | `core/graph.py`, `core/embeddings.py`, `core/scraper.py` |
| Data scraper | `core/scraper.py` — scrapes SHL catalogue (518 assessments) |
| Evaluation | `evaluate.py` — Recall@K on train set + test predictions |
| Test predictions | `output/predictions.csv` (9 queries, 90 rows) |
| Configuration | `config.py` — GPT/Gemini toggle, all settings |
