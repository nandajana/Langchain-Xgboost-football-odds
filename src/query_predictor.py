"""
Single entry point for queries like "Arsenal vs Man City, who wins and what
percentage?". Ties everything together:

  query -> extract team names -> search + fetch data on both teams
        -> estimate structured features from that text (local Ollama model)
        -> run the TRAINED MODEL on those features -> point answer

Key rule, learned the hard way: the LLM NEVER decides which team "wins" --
that comparison is done in plain Python with max(), because a small local
model can misread its own numbers when asked to summarize them in prose.
The LLM's only jobs are (1) estimating input features from search text, and
(2) narrating a winner that Python already determined.

Run: python src/query_predictor.py "Arsenal vs Man City who wins"
"""
import json
import os
import re
import sys

from predict import predict_match
from web_search import search_and_build_documents

# ---------------------------------------------------------------------------
# 1. Parse team names out of a natural-language query
# ---------------------------------------------------------------------------
_SPLIT_PATTERN = re.compile(r"\b(?:vs\.?|versus|v\.?|against)\b", flags=re.IGNORECASE)
_CAP_RUN = re.compile(r"\b(?:[A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*){0,3})\b")
_STOPWORDS = {
    "who", "what", "whats", "predict", "prediction", "probability",
    "chance", "chances", "percentage", "percent", "odds", "win", "wins",
    "winning", "beat", "beats", "by", "and", "do", "does", "they", "will",
    "is", "the", "match",
}


def _pick_team_name(chunk, prefer="last"):
    candidates = [c.strip() for c in _CAP_RUN.findall(chunk) if c.strip()]
    if candidates:
        return candidates[-1] if prefer == "last" else candidates[0]
    words = re.findall(r"[A-Za-z][\w'-]*", chunk)
    run, started = [], False
    for w in words:
        if w.lower() not in _STOPWORDS:
            run.append(w)
            started = True
        elif started:
            break
    return " ".join(run) if run else None


def extract_teams_from_query(query):
    parts = _SPLIT_PATTERN.split(query)
    if len(parts) < 2:
        return None
    home = _pick_team_name(parts[0], prefer="last")
    away = _pick_team_name(parts[1], prefer="first")
    if not home or not away:
        return None
    return home, away


# ---------------------------------------------------------------------------
# 2. Estimate structured features from live search text (local Ollama model)
# ---------------------------------------------------------------------------
FEATURE_EXTRACTION_PROMPT = """You are extracting structured numeric features \
for a football match prediction model, based ONLY on the search text below. \
You are not predicting the outcome -- you are estimating input signals a \
separate statistical model will combine.

Team: {team}
Opponent: {opponent}

SEARCH TEXT:
{text}

Return ONLY a JSON object, no other text, with these fields:
{{
  "elo_estimate": <float 1300-1800, 1500=average top-flight team>,
  "fifa_rating_estimate": <float 50-95, squad quality proxy>,
  "form_pts_estimate": <float 0-15, implied last-5-matches points; 7 if unclear>,
  "goals_avg_estimate": <float 0.5-3.0, implied goals per match recently>,
  "key_absence_penalty": <float 0.0-0.3, injury/suspension impact; 0 if none>
}}"""


def _default_features(team):
    return {"elo_estimate": 1500.0, "fifa_rating_estimate": 70.0,
            "form_pts_estimate": 7.0, "goals_avg_estimate": 1.4,
            "key_absence_penalty": 0.0}


def _call_llm_for_features(team, opponent, text):
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=os.environ.get("OLLAMA_MODEL", "llama3.2"), temperature=0.1)
    prompt = FEATURE_EXTRACTION_PROMPT.format(team=team, opponent=opponent, text=text[:4000])
    raw = llm.invoke(prompt).content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    try:
        return json.loads(match.group(0) if match else raw)
    except (json.JSONDecodeError, AttributeError):
        print(f"  [warn] couldn't parse feature JSON for {team}, using defaults")
        return _default_features(team)


def estimate_team_signals(team, opponent, max_results=4):
    query = f"{team} recent form results injuries squad news vs {opponent}"
    try:
        docs = search_and_build_documents(query, max_results=max_results)
    except Exception as e:
        print(f"  [search failed for {team}: {e}]")
        docs = []

    combined_text = "\n\n".join(d["text"] for d in docs)
    if not combined_text:
        return _default_features(team)
    return _call_llm_for_features(team, opponent, combined_text)


# ---------------------------------------------------------------------------
# 3. Combine into model features and predict
# ---------------------------------------------------------------------------
def build_model_features(home_signals, away_signals):
    home_elo = home_signals["elo_estimate"] * (1 - home_signals.get("key_absence_penalty", 0))
    away_elo = away_signals["elo_estimate"] * (1 - away_signals.get("key_absence_penalty", 0))
    return {
        "home_elo": home_elo, "away_elo": away_elo,
        "home_fifa_rating": home_signals["fifa_rating_estimate"],
        "away_fifa_rating": away_signals["fifa_rating_estimate"],
        "home_form_pts": home_signals["form_pts_estimate"],
        "away_form_pts": away_signals["form_pts_estimate"],
        "home_goals_avg": home_signals["goals_avg_estimate"],
        "away_goals_avg": away_signals["goals_avg_estimate"],
        "h2h_home_win_rate": 0.5,
        "true_home_venue": 1,
    }


def predict_from_query(query, model_path="models/outcome_model.joblib"):
    teams = extract_teams_from_query(query)
    if teams is None:
        return {"error": f"Couldn't identify two teams in query: {query!r}. "
                          f"Try phrasing like 'Arsenal vs Man City'."}
    home, away = teams

    print(f"Parsed matchup: {home} (home) vs {away} (away)")
    print(f"Gathering data on {home}...")
    home_signals = estimate_team_signals(home, away)
    print(f"Gathering data on {away}...")
    away_signals = estimate_team_signals(away, home)

    features = build_model_features(home_signals, away_signals)
    prediction = predict_match(model_path, features)

    # Winner is decided in plain Python -- never by asking the LLM to read
    # the numbers back, since it can misreport which one is actually higher.
    winner = max(prediction, key=lambda k: prediction[k]["probability"])
    winner_prob = prediction[winner]["probability"]
    if winner == "Home win":
        point_answer = f"{home} — {winner_prob:.0%}"
    elif winner == "Away win":
        point_answer = f"{away} — {winner_prob:.0%}"
    else:
        point_answer = f"Draw most likely — {winner_prob:.0%}"

    return {
        "query": query, "home_team": home, "away_team": away,
        "home_signals": home_signals, "away_signals": away_signals,
        "features_used": features, "prediction": prediction,
        "point_answer": point_answer,
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Arsenal vs Man City, who wins and what percentage?"
    out = predict_from_query(q)

    if "error" in out:
        print(out["error"])
        sys.exit(1)

    print(f"\n{'=' * 50}")
    print(f"QUERY: {out['query']}")
    print(f"{out['home_team']} signals: {out['home_signals']}")
    print(f"{out['away_team']} signals: {out['away_signals']}")
    print("\nPREDICTION:")
    for outcome, vals in out["prediction"].items():
        print(f"  {outcome:10s} {vals['probability']:.1%}  (fair odds {vals['fair_decimal_odds']})")
    print(f"\nPOINT ANSWER: {out['point_answer']}")
