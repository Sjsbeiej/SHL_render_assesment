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
            temperature=0.6,
        )
    return _llm


def get_llm_reranker():
    """Deterministic LLM for reranking (temperature=0 for consistency)."""
    global _llm_reranker
    if _llm_reranker is None:
        _llm_reranker = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.5,
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
            enhanced.append(f"{name} {name} {name} {name} {name} {t}")
        _bm25_corpus = [_tokenize(t) for t in enhanced]
        _bm25_index = BM25Okapi(_bm25_corpus)
    return _bm25_index


# ---------------------------------------------------------------------------
# Node 1: Query Analyzer Agent
# ---------------------------------------------------------------------------

QUERY_ANALYZER_PROMPT = """You are an SHL assessment search query generator. Given a job description or hiring query, generate 15-20 search queries to find the best matching SHL assessments.

Your queries feed into BOTH semantic search (FAISS) and keyword search (BM25), so include both styles:
A) KEYWORD QUERIES — include exact words from real SHL product names (for BM25 keyword matching)
B) DESCRIPTIVE QUERIES — natural language descriptions (for FAISS semantic matching)

SHL ASSESSMENT CATALOG — naming patterns to use in keyword queries:

| Category | Naming Pattern | Examples |
|----------|---------------|----------|
| Skill/Knowledge Tests | Named after the skill | "Core Java", "Python", "SQL Server", "Marketing", "Selenium", "Tableau", "Drupal", "HTMLCSS", "CSS3", "Digital Advertising", "Written English", "Data Warehousing" |
| Coding Simulations | "Automata" prefix | "Automata Fix", "Automata SQL", "Automata Selenium", "Automata Pro", "Automata Front End" |
| Writing Simulations | "WriteX" prefix | "WriteX Email Writing Sales", "WriteX Email Writing Managerial" |
| Job-Fit Solutions | Role + version + "JFA/solution" | "Technology Professional 8.0 JFA", "Manager 8.0 JFA", "Professional 7.1 solution", "Professional 7.0 solution" |
| Short Form Packages | Role + "Short Form" | "Administrative Professional Short Form", "Sales Professional Short Form", "Financial Professional Short Form" |
| Pre-packaged Solutions | Role + version/descriptor | "Entry Level Sales 7.1", "Sales Representative Solution", "Director Short Form" |
| Personality | "OPQ" or "Motivation" prefix | "OPQ32", "OPQ Leadership Report", "OPQ Team Types", "Motivation Questionnaire MQM5" |
| Leadership Reports | "Enterprise Leadership" | "Enterprise Leadership Report", "Enterprise Leadership Report 2.0" |
| Cognitive/Verify | "Verify" or "SHL Verify" | "Verify Numerical Ability", "SHL Verify Interactive Numerical Calculation", "Verify Verbal Ability", "SHL Verify Interactive Inductive Reasoning" |
| Communication | Direct name | "Business Communication", "Interpersonal Communications", "SVAR Spoken English", "English Comprehension" |
| Computer/Data | Direct name | "Basic Computer Literacy", "Data Entry", "Microsoft Excel 365", "Microsoft Excel 365 Essentials", "SQL Server Analysis Services SSAS" |
| Global/Broad | "Global Skills" | "Global Skills Assessment", "Global Skills Development Report" |

QUERY GENERATION PRINCIPLES (apply these universally, do NOT hardcode for specific roles):

1. EXPLICIT SKILLS — For every technology, tool, or skill explicitly named in the query, create a keyword query using the closest SHL assessment name from the catalog above.

2. ROLE-LEVEL SOLUTIONS — Identify the seniority and function of the role, then search for matching JFA, Short Form, or pre-packaged solutions:
   - Entry-level → look for "Entry Level ... 7.1" or "... 7.0 solution"
   - Mid-level professional → "Professional 7.1 solution", "Technology Professional 8.0 JFA"
   - Manager/Director → "Manager 8.0 JFA", "Director Short Form"
   - Senior/Executive → "Enterprise Leadership Report", "OPQ Leadership Report"

3. IMPLIED SKILLS — Think about what skills the role REQUIRES even if not explicitly stated:
   - Does it involve numbers/data? → add cognitive/numerical assessments
   - Does it involve writing/communication? → add verbal/communication assessments
   - Does it involve coding? → add relevant Automata simulations
   - Does it involve people/leadership? → add personality/OPQ assessments
   - Does it involve customer interaction? → add communication/interpersonal assessments

4. BREADTH — Cover multiple assessment categories (don't just search for skill tests — also include cognitive, personality, and solution-based assessments that fit the role).

5. ADJACENT SKILLS — Include 2-3 queries for closely related skills the role likely needs (e.g., a Python data role probably also uses SQL and Excel).

6. DESCRIPTIVE QUERIES — Add 3-4 natural language queries describing the role for semantic matching (e.g., "senior data analyst assessment", "leadership and team management evaluation").

7. COMMONLY PAIRED ASSESSMENTS — SHL frequently bundles these general assessments with domain-specific ones. Always include the relevant ones:
   - Roles involving data, budgets, reporting, or analysis → "Microsoft Excel 365 Essentials", "Microsoft Excel 365", "SHL Verify Interactive Numerical Calculation", "Verify Numerical Ability"
   - Roles involving strategy, problem-solving, or decision-making → "SHL Verify Interactive Inductive Reasoning"
   - Roles involving business writing or correspondence → "WriteX Email Writing Sales" or "WriteX Email Writing Managerial"
   - Roles involving spoken communication, customer interaction, or sales → "SVAR Spoken English", "English Comprehension", "Interpersonal Communications"
   - Roles involving data visualization, reporting, or analytics → "Tableau", "Microsoft Excel 365", "Data Warehousing"
   - Roles involving finance or banking → "Financial Professional Short Form", "Verify Numerical Ability"
   - Roles involving coding or programming → "Automata Fix" (code debugging), "Automata Pro" (coding simulation), "Automata SQL" (if SQL is involved)
   - Roles involving web content, CMS, web publishing, or SEO → "Drupal", "Search Engine Optimization"
   - Any professional/mid-level role → "Professional 7.1 solution", "Administrative Professional Short Form"
   - Roles needing broad cognitive screening → "Global Skills Assessment"
   - Any role with verbal/reading requirements → "Verify Verbal Ability"

8. CATEGORY COVERAGE — Try to include queries from at least 4-5 different categories in the catalog table (e.g., skill tests + cognitive/verify + personality + role solutions + communication). Don't concentrate all queries in one category.

IMPORTANT: Always output valid JSON. If the input is a long job description, extract the key role title and skills — do not reproduce the job description in your queries.

Return JSON with:
- "search_queries": list of 15-20 search queries (mix of keyword and descriptive)
- "skills": skills mentioned or implied
- "max_duration_minutes": integer or null (extract from query if mentioned)
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

    url_to_assessment: dict[str, dict] = {}

    # Track per-URL: max score and sum of scores across queries (for both FAISS and BM25)
    faiss_max: dict[str, float] = {}
    faiss_sum: dict[str, float] = {}
    bm25_max: dict[str, float] = {}
    bm25_sum: dict[str, float] = {}
    hit_count: dict[str, int] = {}  # how many queries found this URL

    # Per-query guaranteed slots
    guaranteed_urls: set[str] = set()

    for q_idx in range(len(search_queries)):
        q_vec = query_matrix[q_idx:q_idx + 1]
        scores, indices = index.search(q_vec, top_k)

        # Track top 2 FAISS per query for guaranteed slots
        q_faiss_ranked = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(assessments):
                a = assessments[idx]
                url = a["url"]
                url_to_assessment[url] = a
                s = float(score)
                q_faiss_ranked.append((url, s))
                faiss_max[url] = max(faiss_max.get(url, 0.0), s)
                faiss_sum[url] = faiss_sum.get(url, 0.0) + s
                hit_count[url] = hit_count.get(url, 0) + 1

        # Guarantee top 2 FAISS per query
        for url, _ in q_faiss_ranked[:2]:
            guaranteed_urls.add(url)

    # ---- BM25 keyword search ----
    for sq in search_queries:
        tokens = _tokenize(sq)
        if not tokens:
            continue
        scores = bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        q_bm25_ranked = []
        for idx in top_indices:
            if scores[idx] > 0 and idx < len(assessments):
                a = assessments[idx]
                url = a["url"]
                url_to_assessment[url] = a
                s = float(scores[idx])
                q_bm25_ranked.append((url, s))
                bm25_max[url] = max(bm25_max.get(url, 0.0), s)
                bm25_sum[url] = bm25_sum.get(url, 0.0) + s
                hit_count[url] = hit_count.get(url, 0) + 1

        # Guarantee top 2 BM25 per query
        for url, _ in q_bm25_ranked[:2]:
            guaranteed_urls.add(url)

    # ---- Max+Sum fusion with BM25-only boost ----
    all_urls = set(faiss_max.keys()) | set(bm25_max.keys())

    def _normalize(d: dict[str, float]) -> dict[str, float]:
        if not d:
            return d
        vals = list(d.values())
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {k: 1.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    faiss_max_norm = _normalize(faiss_max)
    bm25_max_norm = _normalize(bm25_max)
    faiss_sum_norm = _normalize(faiss_sum)
    bm25_sum_norm = _normalize(bm25_sum)

    fused_scores: dict[str, float] = {}
    for url in all_urls:
        fm = faiss_max_norm.get(url, 0.0)
        bm = bm25_max_norm.get(url, 0.0)
        fs = faiss_sum_norm.get(url, 0.0)
        bs = bm25_sum_norm.get(url, 0.0)

        # BM25-only boost: strong BM25 + weak FAISS → special scoring
        if bm > 0.3 and fm < 0.1:
            relevance = 0.5 * bm
            breadth = 0.5 * bs
        else:
            relevance = 0.6 * fm + 0.4 * bm
            breadth = 0.6 * fs + 0.4 * bs

        score = 0.7 * relevance + 0.3 * breadth

        # Hit bonus: reward items found by multiple queries
        hits = hit_count.get(url, 1)
        if hits >= 3:
            score += 0.5 * (hits / len(search_queries))

        fused_scores[url] = score

    # Sort by fused score
    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # Build final list: guaranteed slots first (in score order), then rest
    guaranteed_sorted = [(u, fused_scores.get(u, 0)) for u in guaranteed_urls]
    guaranteed_sorted.sort(key=lambda x: x[1], reverse=True)

    final_urls = []
    seen = set()
    # First add top candidates by score
    for url, score in sorted_items:
        if url not in seen:
            seen.add(url)
            final_urls.append((url, score))
        if len(final_urls) >= config.TOP_K_TO_LLM - 10:
            break
    # Then ensure guaranteed slots are included
    for url, score in guaranteed_sorted:
        if url not in seen:
            seen.add(url)
            final_urls.append((url, score))

    final_urls = final_urls[:config.TOP_K_TO_LLM]

    candidates = []
    for url, score in final_urls:
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
        lines.append(line)

    candidates_text = "\n".join(lines)

    max_dur = state.get("max_duration")
    dur_note = f"\n- IMPORTANT: Maximum duration is {max_dur} minutes. Exclude assessments exceeding this." if max_dur else ""

    system_msg = f"""You are an expert SHL assessment consultant. Select exactly {top_k_final} assessments most relevant to the hiring query.

SELECTION PRINCIPLES:

1. NAMED SKILL MATCH (highest priority): If the query names a technology/skill (e.g., "Python", "SQL", "Java", "Selenium", "Excel"), you MUST include the assessment that directly tests that exact skill. EVERY explicitly named skill needs its own test.

2. MAXIMIZE COVERAGE: Fill ALL {top_k_final} slots with assessments that each cover a DIFFERENT aspect of the role. Prefer breadth — don't waste slots on overlapping assessments when uncovered skills remain.

3. ROLE-SPECIFIC PACKAGES: Include role-specific pre-packaged solutions (Short Form, 7.0/7.1 solutions, JFA) that match the role. These are high-value picks:
   - Sales roles → Entry Level Sales, Sales Representative Solution, Technical Sales Associate Solution
   - Admin/banking → Bank Administrative Assistant, Financial Professional Short Form, Entry Level Data Entry
   - Tech roles → Technology Professional 8.0 JFA, Professional 7.1
   - Leadership → Enterprise Leadership Report, Director Short Form
   - Entry-level → look for "Entry Level" or "7.0/7.1 solution" packages

4. PERSONALITY/OPQ ASSESSMENTS:
   - For executive/leadership/COO/VP roles: include MULTIPLE OPQ variants (OPQ32, OPQ Leadership Report, OPQ Team Types) — these roles need deep personality profiling
   - For all other roles: include at most 1 OPQ/personality assessment

5. PRACTICAL SKILLS FOR ENTRY-LEVEL: For entry-level and administrative roles, include practical/operational tests (Basic Computer Literacy, Data Entry, Verify Numerical Ability, Microsoft Excel) rather than abstract cognitive tests.

6. COMMUNICATION-HEAVY ROLES: For roles involving public interaction, media, broadcasting, writing, sales, or customer communication, ALWAYS include verbal/communication tests (English Comprehension, SVAR Spoken English, Interpersonal Communications, Business Communication, Verify Verbal Ability).

7. INCLUDE RELATED TOOLS: For data/analytics roles, include ALL relevant tool-specific tests (SQL Server, SSAS, Excel, Excel Essentials, Tableau, Data Warehousing, Python). For web dev roles, include ALL web technology tests (HTML/CSS, CSS3, JavaScript, etc.).

8. PREFER SPECIFIC OVER GENERIC: "SHL Verify Interactive" over basic "Verify"; "SQL Server" over generic "SQL"; role-specific Short Form over generic Professional solution.
{dur_note}

EXAMPLES of good assessment batteries:

Java Developer (collaboration) → Automata Fix, Core Java Entry Level, Java 8, Core Java Advanced Level, Interpersonal Communications, Technology Professional 8.0 JFA, Automata Pro, OPQ32, Business Communication, Verify Verbal Ability
(Pattern: one test per named skill + coding sims + role-fit JFA)

Entry-level Sales Graduate → Entry Level Sales 7.1, Entry Level Sales Sift Out, Sales Representative Solution, Entry Level Sales Solution, Technical Sales Associate Solution, Business Communication, SVAR Spoken English, Interpersonal Communications, English Comprehension, Verify Verbal Ability
(Pattern: ALL role-specific sales packages + ALL communication/verbal tests)

COO/Executive (cultural fit) → Enterprise Leadership Report, Enterprise Leadership Report 2.0, OPQ Leadership Report, OPQ32, OPQ Team Types, Director Short Form, Global Skills Assessment, Motivation Questionnaire, Manager 8.0 JFA, Verify Interactive Inductive Reasoning
(Pattern: leadership reports + ALL OPQ variants + executive packages + cognitive)

Bank Admin Assistant (entry-level, 30-40 min) → Bank Administrative Assistant Short Form, Administrative Professional Short Form, Financial Professional Short Form, Verify Numerical Ability, Basic Computer Literacy, General Entry Level Data Entry 7.0, Microsoft Excel 365 Essentials, Business Communication, Interpersonal Communications, Data Entry
(Pattern: ALL role-specific packages + practical skills + cognitive)

Content Writer (English + SEO) → Written English, English Comprehension, Search Engine Optimization, Drupal, OPQ32, Verify Verbal Ability, WriteX Email Writing, Business Communication, SVAR Spoken English, Professional 7.1
(Pattern: ALL writing/language tests + domain skills + personality + role-fit)

Radio/Media Station Manager (creative, branding) → Marketing, English Comprehension, Interpersonal Communications, SHL Verify Interactive Inductive Reasoning, Verify Verbal Ability, Manager 8.0 JFA, Digital Advertising, OPQ32, Business Communication, WriteX Email Writing Managerial
(Pattern: domain skills + ALL verbal/comprehension tests + cognitive + management JFA)

QA Engineer (Java, Selenium, SQL, 1hr) → Automata Selenium, Selenium, JavaScript, HTML/CSS, CSS3, SQL Server, Automata SQL, Manual Testing, Professional 7.1, Automata Fix
(Pattern: one test per named technology + ALL related tools + role-fit)

Marketing Manager → Manager 8.0 JFA, Digital Advertising, Marketing, SHL Verify Interactive Inductive Reasoning, WriteX Email Writing Sales, Microsoft Excel 365 Essentials, OPQ32, Business Communication, Interpersonal Communications, Verify Verbal Ability
(Pattern: manager JFA + domain skills + cognitive + writing + personality + communication)

Senior Data Analyst (SQL, Python, Excel, Tableau) → SQL Server, SQL Server Analysis Services SSAS, Automata SQL, Python, Tableau, Microsoft Excel 365, Microsoft Excel 365 Essentials, Data Warehousing, Professional 7.1, Professional 7.0
(Pattern: ALL relevant data tools and technologies + role-fit packages)

Consultant (data analysis, stakeholder management) → SHL Verify Interactive Numerical Calculation, Administrative Professional Short Form, Verify Verbal Ability, OPQ32, Professional 7.1, Microsoft Excel 365, WriteX Email Writing Managerial, Business Communication, SHL Verify Interactive Inductive Reasoning, Interpersonal Communications
(Pattern: cognitive tests + role-fit + personality + writing + ALL communication tests)

KEY PATTERN: Pick ALL available assessments that directly match the role's skills, then fill remaining slots with role-fit packages, cognitive tests, and communication tests.

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
