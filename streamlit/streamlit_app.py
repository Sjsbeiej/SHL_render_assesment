"""
Streamlit frontend for the SHL Assessment Recommendation Engine.
Run from project root: streamlit run streamlit/streamlit_app.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from core.graph import recommend, warmup

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="https://www.shl.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p { color: #b0c4de; margin: 0.5rem 0 0; font-size: 1.05rem; }
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        transition: box-shadow 0.2s;
    }
    .card:hover { box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
    .card-title { font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.3rem; }
    .card-title a { color: #0f3460; text-decoration: none; }
    .card-title a:hover { text-decoration: underline; }
    .card-desc { color: #555; font-size: 0.92rem; line-height: 1.5; }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 0.4rem;
        margin-top: 0.4rem;
    }
    .badge-skill { background: #e8f4fd; color: #0369a1; }
    .badge-sim   { background: #fef3c7; color: #92400e; }
    .badge-pers  { background: #f3e8ff; color: #7c3aed; }
    .badge-apt   { background: #dcfce7; color: #166534; }
    .badge-comp  { background: #ffe4e6; color: #9f1239; }
    .badge-other { background: #f1f5f9; color: #475569; }
    .meta-row { display: flex; gap: 1.5rem; margin-top: 0.6rem; font-size: 0.85rem; color: #64748b; }
    .meta-item { display: flex; align-items: center; gap: 0.3rem; }
    .sample-btn button {
        font-size: 0.82rem !important;
        padding: 0.25rem 0.7rem !important;
    }
    div[data-testid="stExpander"] { border: none !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Badge color mapping
# ---------------------------------------------------------------------------
BADGE_MAP = {
    "Knowledge & Skills": "badge-skill",
    "Simulations": "badge-sim",
    "Personality & Behaviour": "badge-pers",
    "Ability & Aptitude": "badge-apt",
    "Competencies": "badge-comp",
    "Biodata & Situational Judgement": "badge-other",
    "Development & 360": "badge-other",
    "Assessment Exercises": "badge-other",
}

SAMPLE_QUERIES = [
    "I am hiring for Java developers who can also collaborate effectively with my business teams.",
    "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script.",
    "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, the budget is for max 45 min.",
    "I want to hire new graduates for a sales role, the budget is for 30-40 min assessments.",
    "Content Writer required, expert in English and SEO.",
    "I want to hire Customer support executives who are expert in English communication.",
]


# ---------------------------------------------------------------------------
# Load models once
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    warmup()
    return True


with st.spinner("Loading recommendation engine..."):
    load_models()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>SHL Assessment Recommendation Engine</h1>
    <p>Find the most relevant SHL assessments for any role. Enter a job description, skill requirements, or hiring query below.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### How it works")
    st.markdown("""
    1. **Enter** a job description or query
    2. **AI analyzes** the role requirements
    3. **Hybrid search** finds matching assessments
    4. **LLM reranker** selects the top 10

    ---

    **Pipeline:**
    `Query Analyzer` &rarr; `FAISS + BM25 Retriever` &rarr; `LLM Reranker`

    ---
    """)
    st.markdown("### Sample Queries")
    st.caption("Click any to try it out:")

    for sq in SAMPLE_QUERIES:
        label = sq[:60] + "..." if len(sq) > 60 else sq
        if st.button(label, key=sq, use_container_width=True):
            st.session_state["query_input"] = sq


# ---------------------------------------------------------------------------
# Main input
# ---------------------------------------------------------------------------
default_query = st.session_state.get("query_input", "")

query = st.text_area(
    "Job Description / Query",
    value=default_query,
    height=140,
    placeholder="e.g., I am hiring for Java developers who can also collaborate effectively with my business teams.",
    label_visibility="collapsed",
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run = st.button("Get Recommendations", type="primary", use_container_width=True)
with col_info:
    st.caption("Returns up to 10 SHL assessments ranked by relevance")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if run and query.strip():
    # Clear the prefilled query after submission
    if "query_input" in st.session_state:
        del st.session_state["query_input"]

    with st.spinner("Analyzing query and searching assessments..."):
        results = recommend(query)

    if results:
        st.markdown(f"#### Recommended Assessments ({len(results)})")

        for i, r in enumerate(results, 1):
            # Build badges
            badges_html = ""
            for tt in r.get("test_type", []):
                cls = BADGE_MAP.get(tt, "badge-other")
                badges_html += f'<span class="badge {cls}">{tt}</span>'

            # Duration text
            dur = f"{r['duration']} min" if r.get("duration") else "N/A"
            remote = r.get("remote_support", "N/A")
            adaptive = r.get("adaptive_support", "N/A")
            desc = r.get("description", "")[:300]

            st.markdown(f"""
            <div class="card">
                <div class="card-title">{i}. <a href="{r['url']}" target="_blank">{r['name']}</a></div>
                <div class="card-desc">{desc}</div>
                <div>{badges_html}</div>
                <div class="meta-row">
                    <div class="meta-item">&#9202; {dur}</div>
                    <div class="meta-item">&#127760; Remote: {remote}</div>
                    <div class="meta-item">&#9881; Adaptive: {adaptive}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Summary table
        with st.expander("View as Table"):
            table_data = []
            for r in results:
                table_data.append({
                    "Assessment": r["name"],
                    "Duration (min)": r.get("duration", ""),
                    "Remote": r.get("remote_support", ""),
                    "Adaptive": r.get("adaptive_support", ""),
                    "Test Types": ", ".join(r.get("test_type", [])),
                    "URL": r["url"],
                })
            st.dataframe(table_data, use_container_width=True, hide_index=True)

    else:
        st.warning("No recommendations found. Try a different query.")

elif run:
    st.warning("Please enter a query first.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#94a3b8;font-size:0.85rem;">'
    'SHL Assessment Recommendation Engine &middot; Powered by LangGraph + FAISS + BM25 + GPT/Gemini'
    '</div>',
    unsafe_allow_html=True,
)
