from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from pathwise.engine import normalize_profile, recommend, evaluate_profiles
from pathwise.llm import explain

ROOT = Path(__file__).parent
st.set_page_config(page_title="PathWise AI | Personalized Recommendations", page_icon="🧭", layout="wide")
st.markdown("""<style>.block-container{padding-top:2rem;max-width:1250px}div[data-testid='stMetric']{border:1px solid #e2e8f0;padding:12px;border-radius:12px}h1{letter-spacing:-.04em}</style>""", unsafe_allow_html=True)
st.title("🧭 PathWise AI")
st.caption("Personalized learning pathways and education opportunity matching · Research prototype")
st.info("Catalog-based recommendations are demonstrative, not admissions or scholarship eligibility decisions. Always verify details on the provider's website.")

@st.cache_data
def load_catalog(name):
    return pd.read_csv(ROOT / "data" / name).fillna("")

resources = load_catalog("resources.csv")
opportunities = load_catalog("opportunities.csv")
with st.sidebar:
    st.header("Your learner profile")
    goal = st.text_area("What do you want to achieve?", value="Become an NLP engineer and build reliable AI applications", height=100)
    skills = st.text_input("Skills you already have (comma-separated)", "python, pandas, machine learning")
    level = st.selectbox("Current level", ["beginner", "intermediate", "advanced"], index=1)
    hours = st.slider("Study hours per week", 1, 30, 8)
    budget = st.number_input("Maximum resource budget (USD)", min_value=0.0, value=0.0, step=10.0)
    method = st.selectbox("Recommendation approach", ["hybrid", "text-only"], help="Hybrid uses TF-IDF retrieval plus goal, prerequisite, level, time and budget signals. Text-only is a TF-IDF baseline.")
    st.caption("Your profile is used for this session only; no user accounts or persistent profiles are created.")
profile = normalize_profile(goal, skills, hours, budget, level)

def render_cards(frame, label):
    if frame.empty:
        st.warning("No recommendations found.")
        return
    for i, r in frame.iterrows():
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.subheader(f"{i+1}. {r['title']}")
                st.caption(f"{r['provider']} · {r['type']} · {r['level']} · approx. {r['hours']} hours")
                st.write(r["description"])
                st.write("**Why this was suggested:** " + r["reasons"])
                if r["missing_prerequisites"]:
                    st.warning("Prerequisites to review: " + r["missing_prerequisites"])
                st.link_button("Visit official source", r["url"])
            with right:
                st.metric("Ranking score", f"{r['score']:.3f}")
                st.caption("Relative ranking score, not a probability of success")
    st.download_button(f"Download {label} CSV", frame.to_csv(index=False).encode(), file_name=f"pathwise_{label}.csv", mime="text/csv")

learning, opps, evaluation, about = st.tabs(["📚 Learning pathways", "🎯 OpportunityMatch", "📊 Evaluation", "ℹ️ Methodology"])
with learning:
    st.subheader("Recommended learning resources")
    st.write("Ranked against your stated goal, prior skills, learning level, time and budget.")
    ranked = recommend(resources, profile, top_k=7, mode=method)
    render_cards(ranked, "learning")
    st.divider()
    st.subheader("Optional AI explanation")
    st.caption("Gemini explains the catalog-ranked results; it does not select or rerank them.")
    if st.button("Generate personalized explanation", key="learn_explain"):
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")
            model = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
        except (FileNotFoundError, KeyError):
            key, model = "", "gemini-2.5-flash"
        with st.spinner("Preparing evidence-grounded explanation..."):
            output, error = explain(profile, ranked, key, model)
        if output: st.markdown(output)
        if error: st.warning(error)
with opps:
    st.subheader("OpportunityMatch — education and research opportunities")
    st.write("Discover relevant official directories and programs. Listings are illustrative and **not** verified as currently open or individually eligible.")
    ranked_opps = recommend(opportunities, profile, top_k=10, mode=method)
    render_cards(ranked_opps, "opportunities")
    st.caption("Eligibility, geography, deadlines, funding and cohort availability require checking official pages. A zero cost field means no listed catalog access cost; it does not guarantee a program is free.")
with evaluation:
    st.subheader("Reproducible offline ranking evaluation")
    st.write("Compare TF-IDF retrieval against hybrid ranking on **four hand-labeled illustrative scenarios**. These are developer-defined labels, not an independent user study or the AgentRecBench dataset.")
    with open(ROOT / "data" / "evaluation_scenarios.json", encoding="utf-8") as f:
        scenarios = json.load(f)
    report = evaluate_profiles(resources, scenarios)
    st.dataframe(report, use_container_width=True, hide_index=True)
    summary = report.groupby("method", as_index=False)[["precision_at_5", "ndcg_at_5", "budget_satisfaction"]].mean()
    st.subheader("Mean metrics across scenarios")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.bar_chart(summary.set_index("method")[["precision_at_5", "ndcg_at_5"]])
    st.download_button("Download evaluation CSV", report.to_csv(index=False), file_name="pathwise_evaluation.csv", mime="text/csv")
    st.caption("Precision@5 and NDCG@5 depend on manually chosen relevance labels; these results do not establish real-world effectiveness or superiority to an LLM-only baseline.")
with about:
    st.subheader("How the system works")
    st.markdown("""1. **Profile modeling:** Users supply a goal, skills, study time, budget and level.
2. **Retrieval:** TF-IDF unigram/bigram vectorization produces a text similarity score.
3. **Hybrid ranking:** Weighted text relevance, goal-skill overlap, prerequisite readiness, difficulty and study-time fit; over-budget results are penalized.
4. **Evidence-grounded explanations:** Optional Gemini summarizes only the returned catalog entries, without controlling their order.
5. **Evaluation:** A small transparent scenario set compares text-only and hybrid rankings.

**Important:** TF-IDF is lexical vector retrieval, not neural semantic embeddings. This version does not use live web scraping, verified admissions eligibility, persistent learner memory, or an LLM-only baseline. Those are planned research extensions.
""")
    st.write("**Data provenance:** A small hand-curated demonstration catalog of public provider resources. URLs point to official provider pages, but availability and prices may change.")
    st.write("**Privacy:** No authentication or database. Inputs are processed in the current Streamlit session; using Gemini sends the profile and selected catalog items to the model provider.")
