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
from core.embeddings import load_index


def _create_llm(temperature: float):
    """Create LLM instance based on config.LLM_PROVIDER."""
    if config.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
        )
    else:
        return ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=temperature,
        )

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
        _llm = _create_llm(temperature=0.6)
    return _llm


def get_llm_reranker():
    """LLM for reranking with moderate temperature."""
    global _llm_reranker
    if _llm_reranker is None:
        _llm_reranker = _create_llm(temperature=0.5)
    return _llm_reranker


def get_embeddings_model():
    """Always uses OpenAI embeddings regardless of LLM_PROVIDER setting."""
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
| Short Form Packages | Role + "Short Form" | "Administrative Professional Short Form", "Sales Professional Short Form", "Financial Professional Short Form", "Bank Administrative Assistant Short Form", "Professional/Individual Contributor Short Form" |
| Pre-packaged Solutions | Role + version/descriptor | "Entry Level Sales 7.1", "Sales Representative Solution", "Director Short Form", "General Entry Level Data Entry 7.0 Solution" |
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
   - Roles involving data, budgets, reporting, analysis, analytical skills, or ANY manager role → "Microsoft Excel 365 Essentials", "Microsoft Excel 365", "SHL Verify Interactive Numerical Calculation", "Verify Numerical Ability"
   - Roles involving strategy, problem-solving, decision-making, or ANY manager role → "SHL Verify Interactive Inductive Reasoning"
   - Roles involving business writing, correspondence, or ANY manager role → "WriteX Email Writing Sales" or "WriteX Email Writing Managerial"
   - Roles involving spoken communication, customer interaction, or sales → "SVAR Spoken English", "English Comprehension", "Interpersonal Communications"
   - Roles involving data visualization, reporting, or analytics → "Tableau", "Microsoft Excel 365", "Data Warehousing"
   - Roles involving finance, banking, or bank administration → "Financial Professional Short Form", "Bank Administrative Assistant Short Form", "Verify Numerical Ability", "General Entry Level Data Entry 7.0 Solution"
   - Roles involving coding or programming → "Automata Fix" (code debugging), "Automata Pro" (coding simulation), "Automata SQL" (if SQL is involved)
   - Roles involving web content, CMS, web publishing, or SEO → "Drupal", "Search Engine Optimization"
   - Any professional/mid-level role → "Professional 7.1 solution", "Administrative Professional Short Form"
   - Consultant, analyst, or any role asking for cognitive/personality screening → "SHL Verify Interactive Numerical Calculation", "SHL Verify Interactive Inductive Reasoning", "Verify Verbal Ability", "OPQ32", "Professional 7.1 solution", "Administrative Professional Short Form"
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

    # Build numbered candidate list in retriever-ranked order
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

1. NAMED SKILL MATCH (highest priority): If the query names a technology/skill (e.g., "Python", "SQL", "Java", "Excel"), you MUST include the assessment that tests that exact skill. EVERY named skill needs its own test.

2. COMPLETE SKILL COVERAGE: Cover ALL different skill areas. If the query mentions 5 skills, pick 5 different skill tests — don't pick 3 tests for one skill while ignoring others. Breadth over depth.

3. ROLE-FIT SOLUTIONS: Include 1-2 pre-packaged solutions (JFA, Short Form, 7.0/7.1) that match the role type and seniority.

4. LIMIT PERSONALITY TESTS: Pick at most 1 OPQ/personality assessment for most roles. EXCEPTION: For executive/VP/COO/director-level roles focused on cultural fit or leadership, include ALL relevant OPQ and leadership reports (OPQ32, OPQ Leadership Report, OPQ Team Types, etc.) — these roles need multi-faceted personality assessment.

5. LIMIT DUPLICATES: Never pick near-duplicates (e.g., "Professional 7.1 Americas" vs "Professional 7.1 International"). BUT different versions and variants ARE distinct — "Professional 7.0" and "Professional 7.1" are BOTH valid, "Microsoft Excel 365" and "Microsoft Excel 365 Essentials" are BOTH valid, "SQL Server" and "SQL Server Analysis Services SSAS" are BOTH valid. When in doubt, include both variants.

5b. EXHAUST ROLE PACKAGES: When the candidate list has multiple role-specific packages matching the query (e.g., several "Entry Level Sales" variants, several "Administrative" packages), include ALL of them — they test different aspects of the role. Role packages are higher priority than generic cognitive/personality tests.

6. COMMUNICATION-HEAVY ROLES: For roles involving public interaction, media, broadcasting, writing, or customer communication, ALWAYS include verbal/comprehension tests (Verify Verbal Ability, English Comprehension, Interpersonal Communications) alongside domain-specific tests.

7. COGNITIVE SCREENING ROLES: For analyst, consultant, presales, or any role asking for cognitive/personality screening, prioritize ALL cognitive verify tests available (SHL Verify Interactive Numerical Calculation, SHL Verify Interactive Inductive Reasoning, Verify Verbal Ability) + professional solutions (Professional 7.1, Administrative Professional Short Form) + personality (OPQ32).

8. MANAGERS NEED WRITING + ANALYTICS: For any manager role, always include a WriteX email writing simulation and analytical tools (Microsoft Excel 365 Essentials, SHL Verify Interactive Inductive Reasoning) alongside the Manager JFA and domain-specific tests.

9. PREFER SPECIFIC OVER GENERIC: When multiple similar assessments exist, prefer the more specific/advanced one:
   - Prefer "SHL Verify Interactive" versions over basic "Verify" versions (they are more modern)
   - Prefer "SQL Server" over generic "SQL" (more comprehensive)
   - Prefer assessments with "Knowledge & Skills" or "Simulations" types for technical roles over personality-only assessments
{dur_note}

EXAMPLES of good assessment batteries (learn the selection pattern, apply to any role):

React Developer + Node.js, 45 min → ReactJS, JavaScript, Node.js, HTML/CSS, Automata Front End, Automata Fix, Technology Professional 8.0 JFA, Agile Software Development
(Pattern: one test per named skill + coding sims + role-fit JFA)

DevOps Engineer (AWS, Docker, Kubernetes) → AWS Development, Docker, Kubernetes, Linux Administration, Shell Scripting, Cloud Computing, Automata Pro, Professional 7.1
(Pattern: one test per named skill + related technologies + role-fit solution)

VP of Engineering, team building → Enterprise Leadership Report 2.0, Enterprise Leadership 1.0, OPQ32, OPQ Leadership Report, OPQ Team Types, Director Short Form, Global Skills Assessment
(Pattern: leadership reports + personality for cultural fit + executive package)

Hotel Receptionist, fresher → Entry Level Hotel Front Desk Solution, Front Desk Associate Solution, Customer Service Short Form, SVAR Spoken English, English Comprehension, Basic Computer Literacy
(Pattern: role-specific packages + communication + practical skills)

Contact Center Team Lead → Contact Center Manager Short Form, Contact Center Customer Service 8.0, WriteX Email Writing Customer Service, Business Communication, Interpersonal Communications, Verify Verbal Ability, OPQ32
(Pattern: role packages + writing sim + communication + cognitive + personality)

Product Owner (Agile, JIRA, stakeholder management) → Agile Software Development, Project Management, Software Business Analysis, Manager 8.0 JFA, OPQ32, Verify Interactive Inductive Reasoning, WriteX Email Writing Managerial
(Pattern: domain knowledge + manager JFA + personality + reasoning + writing sim)

Bookkeeper, entry-level → Bookkeeping Accounting Auditing Clerk Short Form, Financial Accounting, Accounts Payable, Accounts Receivable, Microsoft Excel 365, Verify Numerical Ability
(Pattern: role package + domain skills + office tools + cognitive aptitude)

Entry-level Sales Graduate, 30-40 min → Entry Level Sales 7.1, Entry Level Sales Sift Out 7.1, Entry Level Sales Solution 7.0, Sales Representative Solution, Technical Sales Associate Solution, Business Communication, SVAR Spoken English (Indian Accent), English Comprehension, Interpersonal Communications, Motivation Questionnaire MQM5
(Pattern: ALL entry-level sales packages first (5 packages) + ALL communication tests. Prioritize role packages over cognitive.)

QA Engineer (Selenium, JavaScript, SQL, HTML/CSS) → Selenium, JavaScript, HTMLCSS, CSS3, SQL Server, Automata Selenium, Automata SQL, Manual Testing, Professional 7.1 solution, Automata Fix
(Pattern: one test per EVERY named skill + ALL matching Automata sims + manual testing + role-fit. Pick skill tests over cognitive tests for technical roles.)

Bank Administrative Assistant, 30-40 min → Bank Administrative Assistant Short Form, Administrative Professional Short Form, Financial Professional Short Form, General Entry Level Data Entry 7.0 Solution, Basic Computer Literacy, Verify Numerical Ability, Microsoft Excel 365, English Comprehension, Data Entry, OPQ32
(Pattern: include ALL matching admin/bank/financial role packages + office skills + cognitive)

Management Consultant, cognitive screening → SHL Verify Interactive Numerical Calculation, Verify Verbal Ability, OPQ32, Professional 7.1 solution, Administrative Professional Short Form, SHL Verify Interactive Inductive Reasoning, Global Skills Assessment, Microsoft Excel 365, Business Communication, WriteX Email Writing Managerial
(Pattern: ALL cognitive verify tests + ALL professional solutions + Administrative Short Form + personality)

Senior Data Analyst (SQL, Python, Tableau) → SQL Server, SQL Server Analysis Services SSAS, Python, Tableau, Data Warehousing, Microsoft Excel 365, Microsoft Excel 365 Essentials, Professional 7.0 solution, Professional 7.1 solution, Automata SQL
(Pattern: ALL skill variants including SSAS + ALL data tools + BOTH Professional 7.0 and 7.1 + coding sim)

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
