from pathlib import Path
import pandas as pd
from pathwise.engine import normalize_profile, recommend, evaluate_profiles

def test_recommendation_is_deterministic():
    data = pd.read_csv(Path(__file__).parents[1] / "data" / "resources.csv").fillna("")
    profile = normalize_profile("NLP transformers", "python", 8, 0, "beginner")
    a = recommend(data, profile)
    b = recommend(data, profile)
    assert a["id"].tolist() == b["id"].tolist()
    assert a["score"].between(0, 1).all()

def test_budget_penalty():
    data = pd.DataFrame([{"id":"x","title":"Python","provider":"X","type":"course","skills":"python","prerequisites":"","level":"beginner","hours":4,"cost_usd":100,"url":"https://example.org","description":"python"}])
    p = normalize_profile("python", "", 8, 0, "beginner")
    assert not bool(recommend(data, p).iloc[0]["eligible_budget"])
