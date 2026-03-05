"""
LangGraph-based SHL Assessment Recommendation Pipeline.

Graph nodes (agents):
1. QueryAnalyzerAgent  - Parses query, extracts skills/requirements, generates search queries
2. RetrieverAgent      - Multi-query FAISS vector search
3. RerankerAgent       - LLM-based re-ranking and final selection
"""

from __future__ import annotations

import json
from typing import TypedDict

import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END

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


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.OPENAI_API_KEY,
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


# ---------------------------------------------------------------------------
# Node 1: Query Analyzer Agent
# ---------------------------------------------------------------------------

QUERY_ANALYZER_PROMPT = """You are an SHL assessment expert. Deeply analyze the hiring query — not just literal words but what the role ACTUALLY requires.

STEP 1 - Break down the query:
- What is the company/industry? (e.g., "ICICI Bank" = banking & finance)
- What is the job role? (e.g., "Assistant Admin" = administrative work, data entry, computer usage)
- What experience level? (e.g., "0-2 years" = entry level)
- What implicit skills does this role need? (e.g., bank admin needs: financial knowledge, numerical skills, computer literacy, data entry)

STEP 2 - Generate search queries covering ALL 4 categories:
1. TECHNICAL/DOMAIN: Tests for explicit AND implied skills from industry+role.
   "ICICI Bank Admin" -> "financial and banking services test", "basic computer literacy test", "data entry assessment"
   "Java developer collaborates" -> "Java programming test", "core Java assessment", "automata coding simulation"
   "Content Writer SEO" -> "SEO knowledge test", "English comprehension test", "written English test"
2. COGNITIVE/APTITUDE: Reasoning tests relevant to the role.
   Banking/finance -> "verify numerical ability test"
   Creative/writing -> "verify verbal ability test"
   Analytical -> "inductive reasoning test"
3. PERSONALITY/BEHAVIORAL: Soft skill assessments.
   "occupational personality questionnaire", "interpersonal communication assessment"
4. ROLE-BASED SOLUTIONS: Pre-built job solutions matching the exact role.
   Bank admin -> "bank administrative assistant short form", "administrative professional short form"
   Sales -> "entry level sales solution", "sales representative solution"
   Manager -> "manager job focused assessment"
   Professional -> "professional job focused assessment"

Return JSON:
- "search_queries": One short query (under 15 words) per skill/requirement. MUST cover ALL 4 categories. Generate 6-10 queries.
- "skills": ALL skills, both explicit AND implied by industry/role
- "max_duration_minutes": integer or null
- "domain": brief domain (e.g., "banking", "software development")"""


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
# Node 2: Retriever Agent
# ---------------------------------------------------------------------------

# Popular generic assessments that apply to most roles — ensures they appear
# as candidates for the reranker even when cosine similarity is low.
POPULAR_ASSESSMENT_KEYWORDS = [
    "occupational personality questionnaire",
    "verify numerical ability",
    "verify verbal ability",
    "interpersonal communication",
    "administrative professional",
    "inductive reasoning",
    "global skills",
    "professional 7.1",
]


def _find_popular_assessments(assessments: list[dict]) -> dict[str, dict]:
    """Find popular generic assessments by name matching."""
    popular = {}
    for a in assessments:
        name_lower = a["name"].lower()
        for keyword in POPULAR_ASSESSMENT_KEYWORDS:
            if keyword in name_lower:
                popular[a["url"]] = a
                break
    return popular


def retriever_node(state: GraphState) -> dict:
    """Multi-query FAISS retrieval — one search per skill/requirement."""
    index, assessments, texts = get_index()
    emb_model = get_embeddings_model()

    search_queries = list(state.get("search_queries", []))
    if not search_queries:
        search_queries = [state["query"][:300]]

    # Add individual skills as search queries (e.g., "Java assessment", "SQL test")
    skills = state.get("skills", [])
    for skill in skills:
        skill_query = f"{skill} assessment test"
        if skill_query not in search_queries:
            search_queries.append(skill_query)

    # Add domain as a search query
    domain = state.get("domain", "")
    if domain:
        domain_query = f"{domain} job assessment solution"
        if domain_query not in search_queries:
            search_queries.append(domain_query)

    top_k = config.TOP_K_RETRIEVAL

    # Embed all skill-based queries at once
    query_vectors = emb_model.embed_documents(search_queries)
    query_matrix = np.array(query_vectors, dtype="float32")
    norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    query_matrix = query_matrix / norms

    # Search per query, keep best score per URL (max-score fusion)
    url_best_score: dict[str, float] = {}
    url_to_assessment: dict[str, dict] = {}

    for q_idx in range(len(search_queries)):
        q_vec = query_matrix[q_idx:q_idx + 1]
        scores, indices = index.search(q_vec, top_k)

        for score, idx in zip(scores[0], indices[0]):
            if idx < len(assessments):
                a = assessments[idx]
                url = a["url"]
                url_to_assessment[url] = a
                if url not in url_best_score or float(score) > url_best_score[url]:
                    url_best_score[url] = float(score)

    # Inject popular generic assessments as candidates
    popular = _find_popular_assessments(assessments)
    for url, a in popular.items():
        if url not in url_best_score:
            url_best_score[url] = 0.30
            url_to_assessment[url] = a

    # Sort by score descending, take top K for re-ranking
    sorted_items = sorted(url_best_score.items(), key=lambda x: x[1], reverse=True)[:config.TOP_K_RERANKER]

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
    dur_note = "\n- IMPORTANT: Maximum duration is {} minutes. Exclude assessments exceeding this.".format(max_dur) if max_dur else ""

    k = config.TOP_K_FINAL
    system_msg = f"""You are an expert SHL assessment recommendation system. Select exactly {k} assessments from the candidates below.

A complete assessment battery MUST include ALL 4 categories:
1. TECHNICAL/DOMAIN (Knowledge & Skills): Pick tests that match the specific skills in the query (e.g., Java, SQL, SEO, Excel). Pick 4-6 of these.
2. COGNITIVE/APTITUDE (Ability & Aptitude): Pick 1-2 reasoning tests (e.g., verify numerical ability, verify verbal ability, inductive reasoning).
3. PERSONALITY/BEHAVIORAL (Personality & Behaviour): Pick 1-2 personality assessments (e.g., OPQ personality questionnaire, interpersonal communication).
4. ROLE SOLUTION: Pick 1 role-based job solution if available (e.g., professional solution, manager JFA, administrative short form).

RULES:
- ALWAYS select exactly {k} assessments covering ALL 4 categories above
- Match specific skills from the query to the right technical tests
- Do NOT pick duplicate/very similar assessments (e.g., don't pick both "Manager 7.1 Americas" and "Manager 7.1 International")
- Prefer assessments with higher similarity scores when relevance is equal{dur_note}

Return JSON: {{"selected": [exactly {k} candidate numbers (1-indexed)]}}"""

    user_msg = f"""Query: {state['query']}

Skills: {state.get('skills', [])}
Domain: {state.get('domain', 'general')}

Candidates:
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
        selected_indices = list(range(1, min(k + 1, len(candidates) + 1)))

    # Map indices to recommendations
    recommendations = []
    seen = set()
    for idx in selected_indices:
        if 1 <= idx <= len(candidates) and candidates[idx - 1]["url"] not in seen:
            seen.add(candidates[idx - 1]["url"])
            recommendations.append(candidates[idx - 1])
        if len(recommendations) >= k:
            break

    # Fill up to k if needed
    for c in candidates:
        if len(recommendations) >= k:
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
