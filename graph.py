"""
LangGraph-based SHL Assessment Recommendation Pipeline.

Graph nodes (agents):
1. QueryAnalyzerAgent  - Parses query, extracts skills/requirements, generates search queries
2. RetrieverAgent      - Multi-query FAISS vector search
3. RerankerAgent       - LLM-based re-ranking and final selection
"""

from __future__ import annotations

import json
import re
from typing import TypedDict

import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from rank_bm25 import BM25Okapi

import config
from embeddings import load_index

# ---------------------------------------------------------------------------
# Shared state flowing through the graph
# ---------------------------------------------------------------------------

class AssessmentCandidate(TypedDict):
    name: str
    url: str
    description: str
    duration: int | None
    remote_support: str
    adaptive_support: str
    test_type: list[str]
    score: float


class GraphState(TypedDict):
    query: str
    search_queries: list[str]
    skills: list[str]
    max_duration: int | None
    domain: str
    candidates: list[AssessmentCandidate]
    recommendations: list[AssessmentCandidate]


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_llm = None
_llm_reranker = None
_embeddings_model = None
_faiss_index = None
_assessments = None
_texts = None
_bm25_index = None
_bm25_corpus = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.0,
        )
    return _llm


def get_llm_reranker():
    """Deterministic LLM for reranking (temperature=0 for consistency)."""
    global _llm_reranker
    if _llm_reranker is None:
        _llm_reranker = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.0,
        )
    return _llm_reranker


def get_embeddings_model():
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )
    return _embeddings_model


def get_index():
    global _faiss_index, _assessments, _texts
    if _faiss_index is None:
        _faiss_index, _assessments, _texts = load_index()
    return _faiss_index, _assessments, _texts


def _tokenize(text: str) -> list[str]:
    """Tokenizer for BM25 with compound word splitting (htmlcss→html+css)."""
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    extra = []
    for t in tokens:
        parts = re.findall(r'[a-z]+|[0-9]+', t)
        if len(parts) > 1:
            extra.extend(p for p in parts if len(p) > 1)
    return tokens + extra


def get_bm25():
    """Build BM25 index with name-boosted corpus (lazy singleton)."""
    global _bm25_index, _bm25_corpus
    if _bm25_index is None:
        _, assessments, texts = get_index()
        enhanced = []
        for a, t in zip(assessments, texts):
            name = a['name'].lower()
            enhanced.append(f"{name} {name} {name} {t}")
        _bm25_corpus = [_tokenize(t) for t in enhanced]
        _bm25_index = BM25Okapi(_bm25_corpus)
    return _bm25_index


# ---------------------------------------------------------------------------
# Node 1: Query Analyzer Agent
# ---------------------------------------------------------------------------

QUERY_ANALYZER_PROMPT = """You generate 12-18 search queries to find SHL assessments. Queries feed into BOTH semantic (FAISS) and keyword (BM25) search, so mix two styles:
A) KEYWORD QUERIES — include exact words from real SHL product names (for BM25 keyword matching)
B) DESCRIPTIVE QUERIES — natural language descriptions (for FAISS semantic matching)

SHL ASSESSMENT CATALOG — naming patterns to use in keyword queries:

| Category | Naming Pattern | Examples |
|----------|---------------|----------|
| Skill/Knowledge Tests | Named after the skill | "Core Java", "Python", "SQL Server", "Marketing", "Selenium" |
| Coding Simulations | "Automata" prefix | "Automata Fix", "Automata SQL", "Automata Selenium" |
| Writing Simulations | "WriteX" prefix | "WriteX Email Writing Sales", "WriteX Email Writing Managerial" |
| Job-Fit Solutions | Role + version + "JFA" | "Technology Professional 8.0 JFA", "Manager 8.0 JFA", "Professional 7.1" |
| Short Form Packages | Role + "Short Form" | "Administrative Professional Short Form", "Financial Professional Short Form" |
| Pre-packaged Solutions | Role + version | "Entry Level Sales 7.1", "Sales Representative Solution" |
| Personality | "OPQ" or "Motivation" prefix | "OPQ32", "OPQ Leadership Report", "OPQ Team Types", "MQM5" |
| Leadership Reports | "Enterprise Leadership" | "Enterprise Leadership Report", "Enterprise Leadership Report 2.0" |
| Cognitive/Verify | "Verify" or "SHL Verify" | "Verify Numerical Ability", "Verify Verbal Ability", "Verify Interactive Inductive Reasoning" |
| Communication | Direct name | "Business Communication", "Interpersonal Communications", "SVAR Spoken English", "English Comprehension" |
| Computer/Data | Direct name | "Basic Computer Literacy", "Data Entry", "Microsoft Excel 365" |
| Global/Broad | "Global Skills" | "Global Skills Assessment", "Global Skills Development Report" |

QUERY GENERATION STRATEGY (apply to ANY role):
1. For each technology/skill NAMED in the query → create a keyword query using the exact SHL assessment name
2. For the role type → find matching JFA, Short Form, or Pre-packaged solutions
3. For supporting skills implied by the role → add relevant cognitive, communication, or personality queries
4. Add 2-3 descriptive/semantic queries about the role

EXAMPLES (showing the principle — generalize to any role):

Query: "Java developer who collaborates, 40 min"
{
  "search_queries": ["Core Java entry level assessment", "Core Java advanced level", "Java 8 programming test", "Automata coding simulation", "Automata Fix code debugging", "technology professional 8.0 job focused assessment", "interpersonal communications skills", "agile software development", "professional 8.0 JFA", "software development knowledge test", "collaboration and teamwork assessment", "object oriented programming"],
  "skills": ["Java", "collaboration", "OOP", "agile"],
  "max_duration_minutes": 40,
  "domain": "software development"
}

Query: "COO for China company, cultural fit, 1 hour"
{
  "search_queries": ["enterprise leadership report", "enterprise leadership report 2.0", "OPQ leadership report", "occupational personality questionnaire OPQ32", "OPQ team types and leadership styles report", "global skills assessment", "executive scenarios narrative report", "motivation questionnaire MQM5", "OPQ emotional intelligence report", "executive short form", "MFS 360 enterprise leadership", "strategic leadership evaluation"],
  "skills": ["operations management", "leadership", "cultural fit", "executive strategy"],
  "max_duration_minutes": 60,
  "domain": "executive leadership"
}

Query: "Senior Data Analyst, SQL, Python, Tableau, 5 years"
{
  "search_queries": ["SQL Server analysis services SSAS", "SQL Server assessment", "Automata SQL coding simulation", "Python programming test", "Tableau data visualization", "data warehousing concepts", "Microsoft Excel 365 essentials", "professional 7.0 solution", "professional 7.1 solution", "technology professional 8.0 job focused", "data science assessment", "numerical reasoning ability test"],
  "skills": ["SQL", "Python", "Tableau", "data analysis", "statistics", "Excel"],
  "max_duration_minutes": null,
  "domain": "data analytics"
}

Query: "ICICI Bank Assistant Admin, 0-2 years, 30-40 min"
{
  "search_queries": ["bank administrative assistant short form", "administrative professional short form", "financial professional short form", "verify numerical ability", "general entry level data entry 7.0 solution", "basic computer literacy Windows 10", "data entry skills assessment", "workplace administration skills", "entry level cashier 7.1", "financial accounting knowledge", "customer service short form"],
  "skills": ["banking", "administration", "data entry", "financial transactions", "computer literacy"],
  "max_duration_minutes": 40,
  "domain": "banking"
}

Return JSON with:
- "search_queries": 12-18 queries using exact SHL assessment name keywords from the catalog table above
- "skills": skills mentioned or implied
- "max_duration_minutes": integer or null
- "domain": brief domain label"""


def query_analyzer_node(state: GraphState) -> dict:
    """Parse the user query and extract structured requirements."""
    llm = get_llm()

    response = llm.invoke([
        {"role": "system", "content": QUERY_ANALYZER_PROMPT},
        {"role": "user", "content": state["query"]},
    ])

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {}

    search_queries = parsed.get("search_queries", [])
    # Always include the raw query as a search query too
    raw_q = state["query"][:300]
    if raw_q not in search_queries:
        search_queries.append(raw_q)

    return {
        "search_queries": search_queries,
        "skills": parsed.get("skills", []),
        "max_duration": parsed.get("max_duration_minutes"),
        "domain": parsed.get("domain", ""),
    }


# ---------------------------------------------------------------------------
# Node 2: Retriever Agent (Hybrid FAISS + BM25)
# ---------------------------------------------------------------------------


def retriever_node(state: GraphState) -> dict:
    """Hybrid retrieval: FAISS semantic + BM25 keyword matching + score fusion."""
    index, assessments, texts = get_index()
    emb_model = get_embeddings_model()
    bm25 = get_bm25()

    search_queries = state.get("search_queries", [])
    if not search_queries:
        search_queries = [state["query"][:300]]

    top_k = config.TOP_K_PER_QUERY

    # ---- FAISS semantic search (per query, max-score fusion) ----
    query_vectors = emb_model.embed_documents(search_queries)
    query_matrix = np.array(query_vectors, dtype="float32")
    norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    query_matrix = query_matrix / norms

    faiss_scores: dict[str, float] = {}
    url_to_assessment: dict[str, dict] = {}

    for q_idx in range(len(search_queries)):
        q_vec = query_matrix[q_idx:q_idx + 1]
        scores, indices = index.search(q_vec, top_k)
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(assessments):
                a = assessments[idx]
                url = a["url"]
                url_to_assessment[url] = a
                if url not in faiss_scores or float(score) > faiss_scores[url]:
                    faiss_scores[url] = float(score)

    # ---- BM25 keyword search (per query, max-score fusion) ----
    bm25_scores: dict[str, float] = {}
    for sq in search_queries:
        tokens = _tokenize(sq)
        if not tokens:
            continue
        scores = bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        for idx in top_indices:
            if scores[idx] > 0 and idx < len(assessments):
                a = assessments[idx]
                url = a["url"]
                url_to_assessment[url] = a
                if url not in bm25_scores or float(scores[idx]) > bm25_scores[url]:
                    bm25_scores[url] = float(scores[idx])

    # ---- Normalize and fuse scores (0.6 FAISS + 0.4 BM25) ----
    all_urls = set(faiss_scores.keys()) | set(bm25_scores.keys())

    # Min-max normalize each score set
    def _normalize(d: dict[str, float]) -> dict[str, float]:
        if not d:
            return d
        vals = list(d.values())
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {k: 1.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    faiss_norm = _normalize(faiss_scores)
    bm25_norm = _normalize(bm25_scores)

    fused_scores: dict[str, float] = {}
    for url in all_urls:
        fs = faiss_norm.get(url, 0.0)
        bs = bm25_norm.get(url, 0.0)
        fused_scores[url] = 0.6 * fs + 0.4 * bs

    # Sort by fused score, take top N
    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:config.TOP_K_TO_LLM]

    candidates = []
    for url, score in sorted_items:
        a = url_to_assessment[url]
        candidates.append(AssessmentCandidate(
            name=a["name"],
            url=a["url"],
            description=a.get("description", ""),
            duration=a.get("duration_minutes"),
            remote_support="Yes" if a.get("remote_testing") else "No",
            adaptive_support="Yes" if a.get("adaptive_irt") else "No",
            test_type=a.get("test_types", []),
            score=score,
        ))

    return {"candidates": candidates}


# ---------------------------------------------------------------------------
# Node 3: Reranker Agent
# ---------------------------------------------------------------------------

def reranker_node(state: GraphState) -> dict:
    """LLM-based re-ranking of retrieved candidates."""
    llm = get_llm_reranker()
    candidates = state["candidates"]
    top_k_final = config.TOP_K_FINAL

    if not candidates:
        return {"recommendations": []}

    # Build numbered candidate list (no scores to avoid bias)
    lines = []
    for i, c in enumerate(candidates, 1):
        line = f"{i}. {c['name']}"
        if c.get("description"):
            line += f" - {c['description'][:200]}"
        line += f" | Types: {', '.join(c.get('test_type', []))}"
        if c.get("duration"):
            line += f" | Duration: {c['duration']}min"
        line += f" | Remote: {c['remote_support']}"
        lines.append(line)

    candidates_text = "\n".join(lines)

    max_dur = state.get("max_duration")
    dur_note = f"\n- IMPORTANT: Maximum duration is {max_dur} minutes. Exclude assessments exceeding this." if max_dur else ""

    system_msg = f"""You are an expert SHL assessment consultant. Select exactly {top_k_final} assessments most relevant to the hiring query.

SELECTION PRINCIPLES:

1. NAMED SKILL MATCH (highest priority): If the query names a technology/skill (e.g., "Python", "SQL", "Java", "Excel"), you MUST include the assessment that tests that exact skill.

2. COMPLETE SKILL COVERAGE: Cover ALL different skill areas in the query. If 5 skills are mentioned, each should have at least one assessment. Don't cluster multiple assessments on one skill while ignoring others.

3. ROLE-FIT SOLUTIONS: Include 1-2 pre-packaged solutions (JFA, Short Form, 7.0/7.1) that match the role type and seniority.

4. NO DUPLICATES: Never pick near-duplicates (e.g., "Professional 7.1 Americas" vs "Professional 7.1 International" — pick one only).
{dur_note}

EXAMPLES of good assessment batteries (learn the selection pattern, apply to any role):

Java Developer + collaboration, 40 min → Core Java Advanced, Core Java Entry Level, Java 8, Automata Fix, Technology Professional JFA, Agile Software Development, Interpersonal Communications
(Pattern: skill-specific tests + coding sim + role-fit JFA + soft skill)

Senior Data Analyst (SQL, Python, Tableau, Excel) → SQL Server Analysis Services, SQL Server, Automata SQL, Python, Tableau, Excel 365 Essentials, Excel 365, Data Warehousing, Professional 7.0, Professional 7.1
(Pattern: one test per named skill + related sim + role-fit solutions)

COO, cultural fit → Enterprise Leadership Report 2.0, Enterprise Leadership 1.0, OPQ32, OPQ Leadership Report, OPQ Team Types, Global Skills Assessment
(Pattern: leadership reports + personality for cultural fit)

Bank Admin, entry-level → Bank Administrative Short Form, Administrative Professional Short Form, Financial Professional Short Form, Verify Numerical, General Entry Level Data Entry, Basic Computer Literacy
(Pattern: role-specific packages + practical aptitude tests)

Sales Graduate → Entry Level Sales 7.1, Sales Sift Out, Sales Representative Solution, Business Communication, SVAR Spoken English, Interpersonal Communications, English Comprehension
(Pattern: role-specific packages + communication + language tests)

Marketing Manager → Marketing, Digital Advertising, SEO, Excel 365 Essentials, Manager 8.0 JFA, WriteX Email Writing, Inductive Reasoning
(Pattern: domain knowledge + manager JFA + writing sim + reasoning)

Consultant → Verify Interactive Numerical, Verify Verbal Ability, OPQ32, Administrative Professional Short Form, Professional 7.1
(Pattern: cognitive tests + personality + role-fit solution)

For any role not shown: apply the same principle — match assessments directly to the skills and competencies the role requires.

Return JSON: {{"selected": [exactly {top_k_final} candidate numbers]}}"""

    skills = state.get("skills", [])
    skills_note = f"\nRequired skills/competencies: {', '.join(skills)}" if skills else ""

    user_msg = f"""Query: {state['query']}
Domain: {state.get('domain', 'general')}{skills_note}

Available assessments (select {top_k_final}):
{candidates_text}"""

    response = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ])

    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        result = json.loads(content)
        selected_indices = result.get("selected", [])
    except (json.JSONDecodeError, AttributeError, IndexError):
        selected_indices = list(range(1, min(top_k_final + 1, len(candidates) + 1)))

    # Map indices to recommendations
    recommendations = []
    seen = set()
    for idx in selected_indices:
        if 1 <= idx <= len(candidates) and candidates[idx - 1]["url"] not in seen:
            seen.add(candidates[idx - 1]["url"])
            recommendations.append(candidates[idx - 1])
        if len(recommendations) >= top_k_final:
            break

    # Fill up if needed
    for c in candidates:
        if len(recommendations) >= top_k_final:
            break
        if c["url"] not in seen:
            seen.add(c["url"])
            recommendations.append(c)

    # Post-filter: enforce duration constraint if specified
    max_dur = state.get("max_duration")
    if max_dur:
        filtered = [r for r in recommendations
                    if not r.get("duration") or r["duration"] <= max_dur]
        # Backfill from candidates that fit within duration
        filtered_urls = {r["url"] for r in filtered}
        for c in candidates:
            if len(filtered) >= top_k_final:
                break
            if c["url"] not in filtered_urls:
                dur = c.get("duration")
                if not dur or dur <= max_dur:
                    filtered_urls.add(c["url"])
                    filtered.append(c)
        recommendations = filtered[:top_k_final]

    return {"recommendations": recommendations}


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct the recommendation pipeline graph."""
    graph = StateGraph(GraphState)

    graph.add_node("query_analyzer", query_analyzer_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reranker", reranker_node)

    graph.set_entry_point("query_analyzer")
    graph.add_edge("query_analyzer", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", END)

    return graph.compile()


# Singleton compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def recommend(query: str) -> list[AssessmentCandidate]:
    """Run the full recommendation pipeline."""
    graph = get_graph()
    initial_state = GraphState(
        query=query,
        search_queries=[],
        skills=[],
        max_duration=None,
        domain="",
        candidates=[],
        recommendations=[],
    )

    result = graph.invoke(initial_state)
    return result["recommendations"]


# Pre-load resources
def warmup():
    """Pre-load FAISS index and models."""
    get_index()
    get_llm()
    get_embeddings_model()


if __name__ == "__main__":
    test_queries = [
        "I am hiring for Java developers who can also collaborate effectively with my business teams.",
        "Looking for a senior data analyst proficient in SQL, Python, and Tableau.",
        "Need a personality and cognitive assessment for entry-level sales roles.",
    ]

    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Query: {query}")
        print("=" * 70)
        results = recommend(query)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['name']} | {', '.join(r['test_type'])} | {r['url']}")
