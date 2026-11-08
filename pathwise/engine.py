"""Deterministic, auditable hybrid recommendation ranking. No model claims are inferred."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tokens(text):
    return set(re.findall(r"[a-z][a-z0-9+#.-]*", str(text).lower()))


def normalize_profile(goal, skills, hours, budget, level):
    return {"goal": str(goal).strip(), "skills": sorted(tokens(skills)), "hours": int(hours), "budget": float(budget), "level": level}


def recommend(df: pd.DataFrame, profile: dict, top_k=8, mode="hybrid") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    frame = df.copy().reset_index(drop=True)
    descriptions = (frame["title"].fillna("") + " " + frame["skills"].fillna("") + " " + frame["description"].fillna(""))
    query = profile["goal"] + " " + " ".join(profile["skills"])
    corpus = descriptions.tolist() + [query]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    vectors = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(vectors[:-1], vectors[-1]).ravel()
    goal_terms = tokens(profile["goal"]) - {"become", "learn", "work", "as", "a", "an", "the", "in", "and", "with", "to"}
    rows = []
    levels = {"beginner": 0, "intermediate": 1, "advanced": 2}
    user_level = levels.get(profile["level"], 0)
    for i, item in frame.iterrows():
        required = tokens(item.get("prerequisites", ""))
        known = set(profile["skills"])
        missing = required - known
        cost = float(item.get("cost_usd", 0) or 0)
        affordable = cost <= profile["budget"]
        match = len(goal_terms & tokens(item.get("skills", ""))) / max(1, len(goal_terms))
        difficulty = levels.get(str(item.get("level", "beginner")).lower(), 0)
        level_fit = max(0.0, 1 - 0.35 * abs(difficulty - user_level))
        readiness = 1.0 if not missing else max(0.0, 1 - len(missing) / max(len(required), 1))
        duration = float(item.get("hours", 10) or 10)
        time_fit = 1.0 if duration <= profile["hours"] * 4 else max(0.1, profile["hours"] * 4 / duration)
        if mode == "text-only":
            score = float(similarity[i])
        else:
            score = (0.42 * float(similarity[i]) + 0.23 * match + 0.17 * readiness + 0.10 * level_fit + 0.08 * time_fit)
            if not affordable:
                score *= 0.2
        reasons = []
        if match:
            reasons.append("Matches goal-related skills")
        if not missing:
            reasons.append("No unmet listed prerequisites")
        else:
            reasons.append("Review prerequisites: " + ", ".join(sorted(missing)))
        if affordable:
            reasons.append("Within stated budget")
        else:
            reasons.append("Exceeds stated budget")
        rows.append({"score": round(float(score), 4), "similarity": round(float(similarity[i]), 4), "goal_overlap": round(match, 3), "missing_prerequisites": ", ".join(sorted(missing)), "eligible_budget": affordable, "reasons": "; ".join(reasons)})
    scored = pd.concat([frame, pd.DataFrame(rows)], axis=1)
    scored = scored.sort_values(["score", "id"], ascending=[False, True]).head(top_k).reset_index(drop=True)
    return scored


def evaluate_profiles(df, scenarios):
    records = []
    for scenario in scenarios:
        profile = normalize_profile(scenario["goal"], scenario["skills"], scenario["hours"], scenario["budget"], scenario["level"])
        relevant = set(scenario["relevant_ids"])
        for method in ["text-only", "hybrid"]:
            results = recommend(df, profile, top_k=5, mode=method)
            ids = results["id"].tolist()
            hits = [int(x in relevant) for x in ids]
            precision = sum(hits) / max(1, len(ids))
            dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
            ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), len(ids))))
            records.append({"scenario": scenario["name"], "method": method, "precision_at_5": round(precision, 3), "ndcg_at_5": round(float(dcg / ideal) if ideal else 0.0, 3), "budget_satisfaction": round(float(results["eligible_budget"].mean()), 3)})
    return pd.DataFrame(records)
