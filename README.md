# PathWise AI — CV-Informed Personalized Career Guidance (Version 3)

**Live application:** https://path-wiseai.streamlit.app/  
**Repository:** https://github.com/UmmeRubab/PathWise_AI

## Research problem

People pursuing career transitions or educational advancement often struggle to translate their qualifications and experience into realistic next steps. Generic guidance may not account for transferable skills, personal goals, local employment opportunities, or accessible ways to develop relevant skills.

## Research goal

Develop and investigate an LLM-powered personalized decision-support system that interprets a CV and a self-defined career goal, identifies transferable skills and development needs, and creates practical, resource-conscious career and learning pathways. The prototype also explores location-aware discovery of public opportunity sources.

## Research question

How effectively can CV-informed profile understanding and goal-directed LLM planning produce relevant, actionable learning and career guidance for individuals with different educational backgrounds, skills, and local opportunities?

## User experience

A single page with three inputs: optional **CV / resume** (text-based PDF, DOCX or TXT, max 5 MB), required **career or education goal**, and optional **city and country**. Results appear in four sections: **Learn Online**, **Jobs Near You / Remote**, **Build Your Skills**, and **Other Opportunities**.

No sidebar, course catalog, raw JSON, numerical fit scores, or chatbot navigation.

## Architecture and implementation

1. Extract text from CV using `pypdf`, `python-docx`, or UTF-8 decoding.
2. Send CV text, user goal and optional location to Gemini to produce a structured plan.
3. Display practical learning activities, realistic job categories, staged skills roadmap, and experience-building opportunities.
4. When a location is supplied, optionally use Gemini Google Search grounding to discover relevant public source pages. Only links supplied in grounding metadata are displayed.
5. If discovery is unavailable, identify roles as suggestions rather than confirmed vacancies and offer an external search link.

**Scope:** This version does not independently verify that a particular vacancy is open, establish eligibility, calculate acceptance probabilities, or train a separate recommendation model. It does not claim these capabilities.

## Deployment

Replace the repository's `app.py`, `requirements.txt`, `README.md`, and `.gitignore`. In Streamlit Community Cloud, set `app.py` as the entry point and add the following under **Secrets**:

```toml
GEMINI_API_KEY = "YOUR_OWN_KEY"
GEMINI_MODEL = "gemini-3.8-flash"
```

Never commit an API key. Google Search grounding depends on provider/model availability and quota.

Windows PowerShell, from the existing PathWise_AI repository folder:

```powershell
git add app.py README.md requirements.txt .gitignore
git commit -m "Redesign PathWise with CV-based personalized plans"
git pull --rebase origin main
git push origin main
```

## Evaluation status

**Research prototype; no formal validation yet.** Future evaluation should compare CV-informed guidance with a goal-only LLM baseline across diverse consented or synthetic CV profiles. Potential metrics: expert-rated relevance, actionability, skill-gap coverage, factual support of opportunity suggestions, location relevance, and fairness across backgrounds. No benchmark gains are claimed.

## Privacy and limitations

- CV content is processed during the session and transmitted to the configured Gemini provider when generating a plan. Provider-side retention is governed by the provider/account terms.
- The app itself does not create a persistent CV database. Avoid uploading national IDs, financial details, or other unnecessary sensitive information.
- Scanned PDFs without selectable text are unsupported; save legacy `.doc` files as `.docx`.
- AI-generated guidance may be inaccurate or incomplete. Source links are discovery aids, not proof of active vacancies.
- The app does not independently verify job eligibility, hiring status, admissions requirements, or program availability.

## Research foundation and references

These sources motivate the approach and proposed evaluation; they do **not** demonstrate this implementation's performance.

1. Xu, Y., et al. (2026). *Toward Personalized LLM-Powered Agents: Foundations, Evaluation, and Future Directions.* arXiv:2602.22680. https://arxiv.org/abs/2602.22680
2. Shang, Y., et al. (2025). *AgentRecBench: Benchmarking LLM Agent-based Personalized Recommender Systems.* NeurIPS 2025 Datasets and Benchmarks. https://arxiv.org/abs/2505.19623
3. *Generative LLMs for educational recommendation.* (2026). *Frontiers in Artificial Intelligence.* https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1827553/full
4. United States Artificial Intelligence Institute (USAII), Certified Artificial Intelligence Engineer (CAIE) curriculum. https://www.usaii.org/
5. Gemini API Google Search grounding documentation. https://ai.google.dev/gemini-api/docs/google-search
6. Streamlit documentation. https://docs.streamlit.io/

## Status

**Version 3 — CV-aware personalized planning prototype.**
