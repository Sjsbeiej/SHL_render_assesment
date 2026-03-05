"""
RAG-based Assessment Recommendation Engine (standalone, not used by main pipeline).
Uses LangChain ChatOpenAI and OpenAIEmbeddings throughout.
"""

import json

import numpy as np
from langchain_openai import ChatOpenAI

import config
from embeddings import load_index, get_embeddings_model

# Lazy-loaded globals
_llm = None
_index = None
_assessments = None
_texts = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0,
            api_key=config.OPENAI_API_KEY,
        )
    return _llm


def load_resources():
    """Load FAISS index and metadata (lazy, singleton)."""
    global _index, _assessments, _texts
    if _index is None:
        _index, _assessments, _texts = load_index()
    return _index, _assessments, _texts


def extract_query_requirements(query: str) -> dict:
    """Use LangChain ChatOpenAI to extract structured requirements from query."""
    llm = get_llm()

    response = llm.invoke([
        {
            "role": "system",
            "content": """You are an expert at analyzing job descriptions and hiring queries to recommend SHL assessments.
Extract structured requirements from the input. Return JSON with these fields:
- "search_queries": list of 3-5 diverse search queries to find relevant assessments
- "skills": list of ALL specific skills mentioned or strongly implied
- "job_level": one of ["entry", "mid", "senior", "manager", "executive", "any"]
- "max_duration_minutes": integer or null if not specified
- "needs_personality": true if personality/behavioral assessment is needed
- "needs_cognitive": true if cognitive/aptitude testing is needed
- "needs_technical": true if specific technical skills need testing
- "domain": brief domain description

Be thorough - extract BOTH explicit and implicit requirements.""",
        },
        {"role": "user", "content": query},
    ])

    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content)
    except (json.JSONDecodeError, AttributeError):
        return {"search_queries": [query], "skills": [], "job_level": "any",
                "max_duration_minutes": None, "needs_personality": True,
                "needs_cognitive": True, "needs_technical": False, "domain": ""}


def multi_query_retrieve(search_queries: list[str], top_k_per_query: int = None) -> list[tuple[dict, float]]:
    """Retrieve candidates using LangChain OpenAIEmbeddings + FAISS."""
    if top_k_per_query is None:
        top_k_per_query = config.TOP_K_RETRIEVAL

    index, assessments, texts = load_resources()
    emb_model = get_embeddings_model()

    # Embed all queries via LangChain
    query_vectors = emb_model.embed_documents(search_queries)
    query_matrix = np.array(query_vectors, dtype="float32")
    norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    query_matrix = query_matrix / norms

    # Search with each query and merge scores (max-score fusion)
    url_best_score = {}
    url_to_assessment = {}

    for q_idx in range(len(search_queries)):
        q_vec = query_matrix[q_idx:q_idx + 1]
        scores, indices = index.search(q_vec, top_k_per_query)

        for score, idx in zip(scores[0], indices[0]):
            if idx < len(assessments):
                url = assessments[idx]["url"]
                url_to_assessment[url] = assessments[idx]
                if url not in url_best_score or score > url_best_score[url]:
                    url_best_score[url] = float(score)

    sorted_candidates = sorted(url_best_score.items(), key=lambda x: x[1], reverse=True)
    return [(url_to_assessment[url], score) for url, score in sorted_candidates]


def rerank_assessments(query: str, requirements: dict, candidates: list[tuple[dict, float]]) -> list[dict]:
    """Use LangChain ChatOpenAI to re-rank and select the best 10 assessments."""
    llm = get_llm()

    candidate_descriptions = []
    for i, (assessment, score) in enumerate(candidates):
        desc = f"{i + 1}. {assessment['name']}"
        if assessment.get("description"):
            desc += f" - {assessment['description'][:200]}"
        desc += f" | Types: {', '.join(assessment.get('test_types', []))}"
        if assessment.get("duration_minutes"):
            desc += f" | Duration: {assessment['duration_minutes']}min"
        desc += f" | Score: {score:.3f}"
        candidate_descriptions.append(desc)

    candidates_text = "\n".join(candidate_descriptions)

    max_dur = requirements.get("max_duration_minutes")
    duration_note = f"\n- IMPORTANT: Maximum duration is {max_dur} minutes." if max_dur else ""

    response = llm.invoke([
        {
            "role": "system",
            "content": f"""You are an expert SHL assessment recommendation system. Select exactly 10 most relevant assessments.

RULES:
1. ALWAYS select exactly 10 assessments
2. BALANCE test types: include BOTH Knowledge/Skills AND Personality/Behaviour when needed
3. Match specific skills from the query{duration_note}

Return JSON: {{"selected": [exactly 10 candidate numbers (1-indexed)]}}""",
        },
        {
            "role": "user",
            "content": f"""Query: {query}

Skills: {requirements.get('skills', [])}
Domain: {requirements.get('domain', 'general')}

Candidates:
{candidates_text}""",
        },
    ])

    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        result = json.loads(content)
        selected_indices = result.get("selected", [])
    except (json.JSONDecodeError, AttributeError):
        selected_indices = list(range(1, min(11, len(candidates) + 1)))

    recommended = []
    seen_urls = set()
    for idx in selected_indices:
        if 1 <= idx <= len(candidates):
            assessment, score = candidates[idx - 1]
            if assessment["url"] not in seen_urls:
                seen_urls.add(assessment["url"])
                recommended.append({
                    "name": assessment["name"],
                    "url": assessment["url"],
                    "description": assessment.get("description", ""),
                    "duration": assessment.get("duration_minutes"),
                    "remote_support": "Yes" if assessment.get("remote_testing") else "No",
                    "adaptive_support": "Yes" if assessment.get("adaptive_irt") else "No",
                    "test_type": assessment.get("test_types", []),
                    "score": score,
                })
        if len(recommended) >= 10:
            break

    # Fill up to 10
    for assessment, score in candidates:
        if len(recommended) >= 10:
            break
        if assessment["url"] not in seen_urls:
            seen_urls.add(assessment["url"])
            recommended.append({
                "name": assessment["name"],
                "url": assessment["url"],
                "description": assessment.get("description", ""),
                "duration": assessment.get("duration_minutes"),
                "remote_support": "Yes" if assessment.get("remote_testing") else "No",
                "adaptive_support": "Yes" if assessment.get("adaptive_irt") else "No",
                "test_type": assessment.get("test_types", []),
                "score": score,
            })

    return recommended


def recommend(query: str) -> list[dict]:
    """Main recommendation function: query → list of recommended assessments."""
    requirements = extract_query_requirements(query)

    search_queries = requirements.get("search_queries", [query])
    if not search_queries:
        search_queries = [query]
    if query[:200] not in search_queries:
        search_queries.append(query[:200])

    candidates = multi_query_retrieve(search_queries, config.TOP_K_RETRIEVAL)
    if not candidates:
        return []

    rerank_candidates = candidates[:50]
    return rerank_assessments(query, requirements, rerank_candidates)


if __name__ == "__main__":
    test_queries = [
        "I am hiring for Java developers who can also collaborate effectively with my business teams.",
        "Looking for a senior data analyst proficient in SQL, Python, and Tableau.",
    ]

    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Query: {query}")
        print("=" * 70)
        results = recommend(query)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['name']} | {', '.join(r['test_type'])} | {r['url']}")
