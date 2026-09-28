import joblib
import pandas as pd

from features import FEATURE_COLUMNS

INV_LABEL_MAP = {0: "Home win", 1: "Draw", 2: "Away win"}


def predict_match(model_path: str, match_features: dict) -> dict:
    model = joblib.load(model_path)
    X = pd.DataFrame([match_features])[FEATURE_COLUMNS]
    probs = model.predict_proba(X)[0]

    result = {}
    for i, p in enumerate(probs):
        label = INV_LABEL_MAP[i]
        result[label] = {
            "probability": round(float(p), 4),
            "fair_decimal_odds": round(1 / p, 2) if p > 0 else None,
        }
    return result


if __name__ == "__main__":
    example = {
        "home_elo": 1620, "away_elo": 1550,
        "home_fifa_rating": 82, "away_fifa_rating": 78,
        "home_form_pts": 11, "away_form_pts": 7,
        "home_goals_avg": 2.1, "away_goals_avg": 1.4,
        "h2h_home_win_rate": 0.6,
        "true_home_venue": 1,
    }
    out = predict_match("models/outcome_model.joblib", example)
    for outcome, vals in out.items():
        print(f"{outcome:10s} prob={vals['probability']:.2%}  fair odds={vals['fair_decimal_odds']}")
