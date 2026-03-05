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
            temperature=0.6,
        )
    return _llm


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
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r'[a-z0-9]+', text.lower())


def get_bm25():
    """Build BM25 index over assessment texts (lazy singleton)."""
    global _bm25_index, _bm25_corpus
    if _bm25_index is None:
        _, assessments, texts = get_index()
        _bm25_corpus = [_tokenize(t) for t in texts]
        _bm25_index = BM25Okapi(_bm25_corpus)
    return _bm25_index


# ---------------------------------------------------------------------------
# Node 1: Query Analyzer Agent
# ---------------------------------------------------------------------------

QUERY_ANALYZER_PROMPT = """You generate search queries to find SHL assessments using BOTH semantic search and keyword matching. Your queries must contain EXACT KEYWORDS that appear in assessment names.

Generate 12-18 diverse search queries. Mix two styles:
A) KEYWORD QUERIES: Use exact words from real SHL product names (for keyword matching)
B) DESCRIPTIVE QUERIES: Use natural language descriptions (for semantic matching)

ROLE-TYPE PATTERNS (from real hiring data — follow these closely):

TECHNICAL ROLES (developer, QA, data analyst, engineer):
  Expected assessments: Knowledge & Skills tests for each named tech + coding simulations + Professional JFA
  Queries must include: exact tech names ("Java", "SQL Server", "Python", "Selenium"), "Automata" for coding sims, "Professional" for JFA
  Example keywords: "Core Java", "Automata Fix", "Automata SQL", "Technology Professional", "Agile Software Development"

SALES ROLES (sales rep, graduates in sales):
  Expected assessments: Entry Level Sales solutions + Sales simulations + Business Communication + Spoken English + Interpersonal
  Queries must include: "entry level sales", "sales sift out", "sales representative", "sales phone simulation", "business communication", "spoken English", "SVAR", "interpersonal", "English comprehension", "sales transformation"

EXECUTIVE/LEADERSHIP ROLES (COO, Director, VP, CEO):
  Expected assessments: Enterprise Leadership Reports + OPQ personality + OPQ Leadership + Team Types + Global Skills
  Queries must include: "enterprise leadership report", "OPQ leadership report", "OPQ team types leadership styles", "personality questionnaire OPQ", "global skills assessment", "executive scenarios"

ADMIN/BANKING ROLES (bank admin, clerk, assistant):
  Expected assessments: Short Form solutions + Verify Numerical + Data Entry + Computer Literacy + Financial services
  Queries must include: "bank administrative assistant", "administrative professional short form", "financial professional", "verify numerical ability", "data entry", "basic computer literacy", "general entry level data entry"

MARKETING/CONTENT ROLES:
  Expected assessments: Domain knowledge + Manager JFA + Excel + Writing simulation + Inductive Reasoning
  Queries must include: "marketing", "digital advertising", "SEO", "Microsoft Excel 365", "manager 8.0 JFA", "WriteX email writing", "inductive reasoning", "interpersonal communications"

CONSULTANT/PROFESSIONAL ROLES:
  Expected assessments: Cognitive tests + OPQ personality + Professional JFA + Administrative Short Form
  Queries must include: "verify interactive numerical calculation", "verify verbal ability", "personality questionnaire OPQ", "administrative professional", "professional 7.1"

EXAMPLES:

Query: "Java developer who collaborates, 40 min"
{
  "search_queries": ["Core Java entry level assessment", "Core Java advanced level", "Java 8 programming test", "Automata coding simulation", "Automata Fix code debugging", "technology professional job focused assessment", "interpersonal communications skills", "agile software development", "professional 8.0 JFA", "software development knowledge test", "coding automation simulation", "collaboration and teamwork assessment", "object oriented programming"],
  "skills": ["Java", "collaboration", "OOP", "agile"],
  "max_duration_minutes": 40,
  "domain": "software development"
}

Query: "Sales graduates, budget for 3 assessments under 30 min"
{
  "search_queries": ["entry level sales 7.1 solution", "entry level sales sift out", "entry level sales solution", "sales representative solution", "sales and service phone simulation", "business communication adaptive", "SVAR spoken English Indian accent", "interpersonal communications", "English comprehension", "graduate 8.0 job focused assessment", "sales transformation individual contributor", "technical sales associate solution", "sales profiler cards", "general entry level all industries", "customer engagement assessment"],
  "skills": ["sales", "customer engagement", "communication", "English"],
  "max_duration_minutes": 30,
  "domain": "sales"
}

Query: "COO for China company, cultural fit, 1 hour"
{
  "search_queries": ["enterprise leadership report", "enterprise leadership report 2.0", "OPQ leadership report", "occupational personality questionnaire OPQ32", "OPQ team types and leadership styles report", "global skills assessment", "executive scenarios narrative report", "motivation questionnaire MQM5", "OPQ emotional intelligence report", "executive short form", "director short form", "managerial scenarios", "MFS 360 enterprise leadership", "strategic leadership evaluation"],
  "skills": ["operations management", "leadership", "cultural fit", "executive strategy"],
  "max_duration_minutes": 60,
  "domain": "executive leadership"
}

Query: "Marketing Manager, brand positioning, digital campaigns, content strategy"
{
  "search_queries": ["marketing knowledge assessment", "digital advertising test", "SEO search engine optimization", "Microsoft Excel 365 essentials", "manager 8.0 JFA job focused", "manager 8.0+ JFA", "WriteX email writing sales", "SHL verify interactive inductive reasoning", "interpersonal communications", "business communications", "professional 8.0 JFA", "brand management assessment", "content strategy writing skills", "manager short form"],
  "skills": ["marketing", "digital advertising", "SEO", "content strategy", "Excel"],
  "max_duration_minutes": null,
  "domain": "marketing"
}

Query: "Senior Data Analyst, SQL, Python, Tableau, 5 years"
{
  "search_queries": ["SQL Server analysis services SSAS", "SQL Server assessment", "Automata SQL coding simulation", "Python programming test", "Tableau data visualization", "data warehousing concepts", "Microsoft Excel 365", "Microsoft Excel 365 essentials", "data science assessment", "professional 7.0 solution", "professional 7.1 solution", "technology professional 8.0 job focused", "basic statistics assessment", "numerical reasoning ability test", "machine learning assessment"],
  "skills": ["SQL", "Python", "Tableau", "data analysis", "statistics", "Excel"],
  "max_duration_minutes": null,
  "domain": "data analytics"
}

Query: "ICICI Bank Assistant Admin, 0-2 years, 30-40 min"
{
  "search_queries": ["bank administrative assistant short form", "administrative professional short form", "financial professional short form", "verify numerical ability", "general entry level data entry 7.0 solution", "basic computer literacy Windows 10", "financial and banking services", "data entry skills assessment", "workplace administration skills", "entry level cashier 7.1", "financial accounting knowledge", "accounts payable assessment", "customer service short form", "personal banker short form"],
  "skills": ["banking", "administration", "data entry", "financial transactions", "computer literacy"],
  "max_duration_minutes": 40,
  "domain": "banking"
}

Query: "Consultant position, assessment under 40 min"
{
  "search_queries": ["SHL verify interactive numerical calculation", "verify verbal ability next generation", "occupational personality questionnaire OPQ32", "administrative professional short form", "professional 7.1 solution international", "professional 7.0 solution", "inductive reasoning test", "deductive reasoning assessment", "business communication skills", "interpersonal communications", "motivation questionnaire MQM5", "manager short form", "graduate 8.0 job focused assessment"],
  "skills": ["analytical thinking", "communication", "problem solving", "consulting"],
  "max_duration_minutes": 40,
  "domain": "consulting"
}

Return JSON with:
- "search_queries": 12-18 queries. Include EXACT KEYWORDS from real SHL assessment names (e.g., "Automata", "OPQ32", "verify numerical ability", "short form", "JFA", "7.1", "8.0").
- "skills": list of skills mentioned or implied
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
    llm = get_llm()
    candidates = state["candidates"]

    if not candidates:
        return {"recommendations": []}

    # Build numbered candidate list
    lines = []
    for i, c in enumerate(candidates, 1):
        line = f"{i}. {c['name']}"
        if c.get("description"):
            line += f" - {c['description'][:200]}"
        line += f" | Types: {', '.join(c.get('test_type', []))}"
        if c.get("duration"):
            line += f" | Duration: {c['duration']}min"
        line += f" | Remote: {c['remote_support']}"
        line += f" | Score: {c['score']:.3f}"
        lines.append(line)

    candidates_text = "\n".join(lines)

    max_dur = state.get("max_duration")
    dur_note = f"\n- IMPORTANT: Maximum duration is {max_dur} minutes. Exclude assessments exceeding this." if max_dur else ""

    top_k_final = config.TOP_K_FINAL

    system_msg = f"""You are a senior SHL assessment consultant. Re-rank the candidate assessments by relevance to the hiring query and pick the top {top_k_final}.

SCORING GUIDE — What makes an assessment relevant depends on the ROLE TYPE:

FOR TECHNICAL ROLES (developer, QA, data analyst, engineer):
- HIGHEST: Knowledge & Skills tests that directly match named technologies (Java 8, Python, SQL Server, Selenium, HTML/CSS)
- HIGH: Coding simulations (Automata, Automata Fix, Automata SQL, Automata Selenium)
- MEDIUM: Job-fit solutions (Technology Professional JFA, Professional 7.1)
- LOW: Generic personality/cognitive tests (unless query specifically mentions collaboration or reasoning)

FOR SALES ROLES (sales rep, sales graduate):
- HIGHEST: Pre-packaged Sales Solutions (Entry Level Sales 7.1, Sales Sift Out, Sales Representative Solution)
- HIGH: Sales simulations (Sales & Service Phone Simulation), communication tests (Business Communication, SVAR Spoken English)
- MEDIUM: Interpersonal Communications, English Comprehension
- LOW: Cognitive/personality tests (unless query asks for them)

FOR EXECUTIVE/LEADERSHIP ROLES (COO, Director, VP):
- HIGHEST: Leadership reports (Enterprise Leadership Report 1.0/2.0, OPQ Leadership Report)
- HIGH: Personality assessments (OPQ32, OPQ Team Types and Leadership Styles)
- MEDIUM: Executive solutions (Executive Short Form), Global Skills Assessment
- LOW: Technical knowledge tests

FOR ADMIN/BANKING ROLES (bank admin, clerk, assistant):
- HIGHEST: Pre-packaged Short Forms (Bank Administrative Assistant, Administrative Professional, Financial Professional)
- HIGH: Verify Numerical Ability, Data Entry, Basic Computer Literacy
- MEDIUM: General Entry Level solutions, Workplace Administration Skills
- LOW: Advanced technical or leadership tests

FOR MARKETING/CONTENT ROLES:
- HIGHEST: Domain knowledge (Marketing, Digital Advertising, SEO, Excel)
- HIGH: Manager solutions (Manager 8.0+ JFA), Writing simulations (WriteX Email Writing)
- MEDIUM: Inductive Reasoning, Interpersonal Communications
- LOW: Unrelated technical tests

GENERAL RULES:
- Assessments matching a NAMED skill in the query (e.g., "SQL" → SQL Server test) are always top priority
- Pre-packaged solutions (JFA, Short Form, 7.1) matching the role type are highly relevant
- NEVER pick duplicates or near-duplicates (e.g., don't pick both "7.1 Americas" and "7.1 International")
- Prefer assessments with higher retrieval scores when relevance is equal{dur_note}

Select the top {top_k_final} most relevant. Return JSON: {{"selected": [exactly {top_k_final} candidate numbers (1-indexed)]}}"""

    user_msg = f"""I need to hire for this role. Design the assessment battery.

Role/Query: {state['query']}

Key skills to test: {state.get('skills', [])}
Domain: {state.get('domain', 'general')}

Available assessments (pick {top_k_final}):
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
