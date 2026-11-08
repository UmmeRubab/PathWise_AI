# PathWise AI

**Personalized Learning Pathways and Education Opportunity Matching**

PathWise AI is a Streamlit research prototype that helps learners explore learning resources and educational opportunities based on their goals, existing skills, available study time, budget, and experience level. Its **OpportunityMatch** feature provides a separate view of official learning, fellowship, scholarship-directory, and research-opportunity resources.

> **Research question:** Can a hybrid recommendation system combining structured learner profiles, vector-based retrieval, and evidence-grounded LLM explanations produce more relevant and reliable learning recommendations than simpler baselines?
>
> This question is a research motivation, **not an established experimental result**. The current release compares hybrid ranking with a lexical retrieval baseline; an LLM-only comparison and independent user study are future work.

## Research problem

Learners and students often struggle to identify educational opportunities, relevant skills, and learning resources that align with their academic background, career aspirations, financial constraints, and available time. Existing guidance platforms may provide broad recommendations without sufficiently adapting to each individual's circumstances, evolving goals, and learning progress.

PathWise AI explores how personalized, accessible guidance can connect learners with relevant resources and educational opportunities. It does **not** claim to determine admission or scholarship eligibility.

## Research goal and objectives

**Primary goal:** Develop and evaluate an intelligent, LLM-powered personalized recommendation system that analyzes learner profiles, identifies skill gaps, and recommends suitable learning pathways, educational resources, and academic opportunities. The system aims to make educational and career guidance more accessible, adaptive, and actionable through natural language interaction, semantic matching, and explainable recommendations.

The project aims to:

1. **Represent learner needs** through goals, existing skills, experience level, weekly study time, and budget.
2. **Rank learning resources transparently** using relevance and explicit learner constraints, rather than relying solely on free-form LLM suggestions.
3. **Explain recommendations with traceable catalog information** and clearly distinguish resource discovery from verified opportunity availability or eligibility.
4. **Evaluate recommendation quality** against a simpler retrieval baseline using relevance-ranking and constraint-satisfaction metrics.
5. **Investigate future improvements**, including neural semantic retrieval, cold-start personalization, changing learner preferences, and an LLM-only comparison.

**Research hypothesis (to be tested):** A profile-aware hybrid ranking approach can improve relevance and constraint satisfaction compared with a text-only retrieval baseline, while grounded LLM explanations may improve interpretability without changing the underlying ranking. This is a hypothesis, not a demonstrated outcome.

**Scope of version 1.0:** The current implementation uses a curated catalog, TF-IDF retrieval, hand-designed ranking weights, and four illustrative evaluation scenarios. It is a proof of concept rather than a validated large-scale recommender. The optional Gemini layer generates explanations; it does not validate opportunities or make eligibility decisions.

## Live application

Deploy this repository on Streamlit Community Cloud with `app.py` as the entry point. No live URL is claimed until deployment is complete.

## Features implemented in version 1.0

- **Learner profile:** Goal, skills, experience level, weekly study time, and budget.
- **Learning recommendations:** Ranked catalog of public learning resources, with score components and understandable reasons.
- **OpportunityMatch:** Relevant official program and opportunity-directory links, clearly distinguished from confirmed open positions or eligibility decisions.
- **Auditable hybrid ranking:** TF-IDF lexical vector retrieval plus explicit goal overlap, prerequisite readiness, level fit, time fit, and budget penalty.
- **Optional Gemini explanations:** The LLM explains **already-ranked** resources using supplied catalog facts; it does not control ranking or verify eligibility.
- **Evaluation dashboard:** Text-only retrieval versus hybrid ranking using Precision@5, NDCG@5, and budget-satisfaction rate on four hand-labeled illustrative scenarios.
- **CSV exports:** Ranked resources, opportunities, and offline evaluation results.
- **API-key-free mode:** Ranking, opportunity matching, and evaluation work without Gemini.

## Architecture

```text
User goal + skills + level + weekly time + budget
                        |
                Structured profile
                        |
           Curated catalog (CSV files)
                        |
        TF-IDF unigram/bigram vectorization
                        |
          Text similarity (cosine score)
                        |
     Hybrid constraints and relevance scoring
                        |
            Ranked recommendations
                 /              \
     Explainable UI       Optional Gemini
     + official links     grounded explanation
                        |
             Offline baseline evaluation
```

### Ranking methodology

For each candidate resource, the hybrid ranking score is:

```text
score = 0.42 * TF-IDF cosine similarity
      + 0.23 * goal-to-skill token overlap
      + 0.17 * prerequisite readiness
      + 0.10 * learner-level fit
      + 0.08 * time fit
```

An item that exceeds the stated budget receives a multiplicative penalty of 0.2. Scores are **relative ranking signals**, not calibrated probabilities of success, suitability, or acceptance. The text-only baseline uses TF-IDF cosine similarity alone. Weight values are initial engineering choices, not experimentally optimized coefficients.

**Terminology:** TF-IDF is lexical vector retrieval; the initial release does **not** implement neural semantic embeddings, a trained recommendation model, or a persistent user-memory agent.

## Technology

- Python 3.12 recommended
- Streamlit
- pandas, NumPy, scikit-learn
- Google Gemini via `google-genai` (optional explanation layer)
- CSV catalogs and JSON evaluation scenarios

## Local installation (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

A Gemini key is **optional**. To enable explanations, create `.streamlit/secrets.toml` locally:

```toml
GEMINI_API_KEY = "YOUR_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash"
```

The secrets file is excluded by `.gitignore`. Never commit an API key. Model availability, quotas, and pricing are controlled by the provider; set a model accessible to your account.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and push the project files (do not commit `.venv` or `.streamlit/secrets.toml`).
2. In Streamlit Community Cloud, select the repository and `app.py`.
3. Add `GEMINI_API_KEY` and `GEMINI_MODEL` under the application's **Secrets** settings only if optional explanations are desired.
4. Deploy and test the learning, opportunity, and evaluation tabs.

## Repository structure

```text
PathWise_AI/
├── app.py
├── pathwise/
│   ├── __init__.py
│   ├── engine.py
│   └── llm.py
├── data/
│   ├── resources.csv
│   ├── opportunities.csv
│   └── evaluation_scenarios.json
├── tests/
│   └── test_engine.py
├── .streamlit/
│   └── config.toml
├── .gitignore
├── requirements.txt
└── README.md
```

## Evaluation design and limitations

The `Evaluation` tab compares the text-only and hybrid ranking methods on four **developer-created illustrative profiles** with manually selected relevant catalog IDs. It reports Precision@5, NDCG@5, and the fraction of returned items within the specified budget. These labels are not independently annotated, and the small catalog and scenario count are insufficient for claims about statistical significance, fairness, real learner outcomes, or superior performance against a generative LLM.

**Next research steps:**

1. Add neural embeddings and measure gains over TF-IDF.
2. Implement a reproducible LLM-only baseline and an ablation study for each hybrid feature.
3. Expand to independently labeled recommendation relevance judgments, cold-start and changing-preference scenarios, and larger datasets.
4. Track learning progress with privacy-conscious, opt-in profile memory.
5. Add an opportunity ingestion pipeline with source timestamps, eligibility parsing, provenance, and freshness checks.
6. Assess explanation faithfulness separately from recommendation relevance and investigate demographic fairness and constraint violations.

## OpportunityMatch safety and data provenance

The included catalog is a small **hand-curated demonstration dataset** pointing to public provider pages. Entries are not live scraped and should **not** be treated as currently open scholarships, fellowships, admissions offers, or verified individual eligibility. Program rules, prices, cohort dates, country restrictions, and deadlines can change. Some entries are directories rather than individual opportunities. A zero catalog cost does not imply that all linked programs are free.

The application does not create accounts or persist learner profiles. When optional Gemini explanations are used, the learner profile and selected resource descriptions are transmitted to the configured model provider. Users should avoid entering sensitive personal information.

## Research foundation and references

The following works inform the research motivation and future evaluation design; **citing them does not imply their full algorithms have been implemented**.

1. **Xu, Y., et al. (2026).** *Toward Personalized LLM-Powered Agents: Foundations, Evaluation, and Future Directions.* arXiv:2602.22680. https://arxiv.org/abs/2602.22680 — Surveys personalized LLM agents through profile modeling, memory, planning, and action execution.
2. **Shang, Y., et al. (2025).** *AgentRecBench: Benchmarking LLM Agent-based Personalized Recommender Systems.* Advances in Neural Information Processing Systems (NeurIPS 2025), Datasets and Benchmarks Track. https://arxiv.org/abs/2505.19623 ; https://proceedings.neurips.cc/paper_files/paper/2025/hash/e2d6f7249add096e26679eade1b4cc6f-Abstract-Datasets_and_Benchmarks_Track.html — Introduces classic, cold-start, and evolving-interest recommendation evaluation settings.
3. **Generative LLMs for educational recommendation (2026).** *Frontiers in Artificial Intelligence.* https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1827553/full — Discusses deterministic educational ranking with evidence-grounded generative explanations and evaluation of explanation reliability.
4. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html — Foundation for the vectorization and similarity implementation.

## Status

**Version 1.0 — Implemented research prototype.** The software is suitable for demonstrating a reproducible hybrid ranking pipeline and identifying questions for more rigorous recommendation research. It has not been validated as a production admissions adviser or an experimentally proven personalized learning intervention.
