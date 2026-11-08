"""PathWise AI: personalized education and career decision support."""
from __future__ import annotations

import os
from datetime import date
from urllib.parse import urlparse

import streamlit as st

st.set_page_config(page_title="PathWise AI | Your Personal AI Advisor", page_icon="🧭", layout="centered")
st.markdown("""<style>
.block-container {max-width: 940px; padding-top: 2.2rem; padding-bottom: 4rem}
h1 {letter-spacing: -.045em; margin-bottom: .1rem}
[data-testid="stSidebar"] {background: #f7faf9}
.stButton > button {border-radius: 12px}
[data-testid="stChatMessage"] {border-radius: 14px; padding: 1rem}
</style>""", unsafe_allow_html=True)

SYSTEM = """You are PathWise AI, a thoughtful personalized education and career advisor.
Your mission is to help people make practical education and career decisions, especially learners with nontraditional backgrounds and limited resources.
Use the supplied user profile and prior conversation to tailor your advice. Never assume unstated degrees, citizenship, grades, finances, or experience.
For each request: understand the goal; identify relevant transferable strengths and gaps; give 2-4 realistic paths or options with trade-offs; recommend concrete next actions. Keep responses easy to read and avoid unnecessary technical jargon.
Ask at most one essential clarifying question if it would substantially change the recommendation; otherwise proceed with transparent assumptions.
For academic programs, scholarships, funding, deadlines, immigration, or eligibility: do not invent institutions, awards, amounts, opening status, URLs, requirements, or guarantees. If current official sources have not been verified, say 'Current details have not been verified; check the official provider website.' Distinguish likely fit from formal eligibility. Do not invent citations.
For learning plans, recommend skills, milestones and projects tailored to the user's constraints; do not default to course catalogs or commercial course promotions.
Do not expose raw reasoning, model internals, or numeric similarity scores. Never claim that the user's information is saved across sessions.
"""

STARTERS = [
    "I want to change careers. Help me explore realistic options.",
    "Help me find suitable higher-education and scholarship opportunities.",
    "Create a personalized action plan for my education or career goal.",
    "I have a nontraditional academic background. What are my next steps?",
]


def secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return os.getenv(name, default)


def safe_sources(response):
    """Only display URLs supplied by Google's grounding metadata."""
    output = []
    try:
        candidates = response.candidates or []
        for candidate in candidates:
            metadata = getattr(candidate, "grounding_metadata", None)
            for chunk in (getattr(metadata, "grounding_chunks", None) or []):
                web = getattr(chunk, "web", None)
                uri = getattr(web, "uri", None)
                title = getattr(web, "title", None)
                if uri and urlparse(uri).scheme == "https" and uri not in [x[1] for x in output]:
                    output.append((str(title or urlparse(uri).netloc), str(uri)))
    except Exception:
        pass
    return output[:8]


def ask_gemini(messages, profile, use_search):
    from google import genai
    from google.genai import types

    api_key = secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Add GEMINI_API_KEY in your Streamlit app's Secrets settings to enable AI guidance.")
    model = secret("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)
    context = "User-provided profile (blank means unknown):\n" + "\n".join(f"{k}: {v or 'Not provided'}" for k, v in profile.items())
    context += f"\nToday's date: {date.today().isoformat()}\n"
    recent = messages[-14:]
    transcript = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
    prompt = context + "\nConversation:\n" + transcript + "\n\nRespond to the latest USER message."
    config_args = {"system_instruction": SYSTEM, "temperature": 0.35}
    if use_search:
        config_args["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    response = client.models.generate_content(model=model, contents=prompt, config=types.GenerateContentConfig(**config_args))
    answer = (response.text or "").strip()
    if not answer:
        raise RuntimeError("The AI service returned no text. Please try again.")
    return answer, safe_sources(response)


st.title("🧭 PathWise AI")
st.caption("Personalized education, career guidance and opportunity discovery")
st.write("Tell me where you are today and where you want to go. I'll help you explore your options and decide what to do next.")

with st.sidebar:
    st.subheader("Your background (optional)")
    education = st.text_area("Education", placeholder="e.g. Bachelor's in biology", height=75)
    experience = st.text_area("Experience and skills", placeholder="e.g. 2 years in banking, Python basics", height=90)
    location = st.text_input("Country or preferred destination", placeholder="e.g. Pakistan; study in Germany")
    constraints = st.text_area("Budget, time or other preferences", placeholder="e.g. Need funding; 6 hours/week", height=85)
    st.caption("This information is used for the current session. Avoid entering sensitive personal details.")
    if st.button("Start new conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

profile = {"Education": education, "Experience and skills": experience, "Location preferences": location, "Constraints": constraints}
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("**What can I help you with?**")
    for i, starter in enumerate(STARTERS):
        if st.button(starter, key=f"starter_{i}", use_container_width=True):
            st.session_state.pending = starter
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources found during this response"):
                for title, url in m["sources"]:
                    st.markdown(f"- [{title}]({url})")

with st.expander("Search preferences", expanded=False):
    use_search = st.checkbox("Use Google Search grounding for current opportunities and program details", value=True, help="Uses supported Gemini grounding; availability depends on the selected model and API account.")

question = st.chat_input("Ask about your education, career or future opportunities...")
question = question or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Preparing your personalized guidance..."):
            try:
                answer, sources = ask_gemini(st.session_state.messages, profile, use_search)
                st.markdown(answer)
                if sources:
                    with st.expander("Sources found during this response"):
                        for title, url in sources:
                            st.markdown(f"- [{title}]({url})")
                elif use_search:
                    st.caption("No source links were returned. Current opportunity details are not independently verified.")
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as exc:
                st.error(f"AI guidance unavailable: {exc}")
                st.info("Check GEMINI_API_KEY, GEMINI_MODEL, and API quota in Streamlit Secrets. If your model does not support search grounding, turn off the search option and retry.")
                st.session_state.messages.pop()

st.divider()
st.caption("PathWise provides guidance, not official admission, scholarship, immigration or employment decisions. Verify time-sensitive requirements with official organizations.")
