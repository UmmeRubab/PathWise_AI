"""Optional Gemini explanations; ranked results remain determined by engine.py."""
import os

def explain(profile, results, api_key, model):
    if not api_key:
        return None, "Add GEMINI_API_KEY in Streamlit secrets to enable optional AI explanations."
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        items = []
        for _, r in results.head(5).iterrows():
            items.append({"id": str(r["id"]), "title": str(r["title"]), "skills": str(r["skills"]), "prerequisites": str(r["prerequisites"]), "reasons": str(r["reasons"]), "url": str(r["url"])})
        prompt = ("You are an educational recommendation explainer, NOT a ranking engine. "
                  "Explain ONLY these provided recommendations. Never invent courses, deadlines, scholarships, "
                  "eligibility, guaranteed outcomes, or claims about the learner. Cite resource IDs in brackets. "
                  "Keep under 230 words. Clearly say opportunity availability and eligibility must be checked at source. "
                  f"Learner profile: {profile}\nRanked catalog items: {items}")
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text, None
    except Exception as exc:
        return None, f"AI explanation unavailable ({type(exc).__name__}). Recommendations remain available."
