# Football Match Outcome Predictor — XGBoost + LangChain RAG

Predicts Win/Draw/Loss probabilities for a football match from multi-factor
team data (Elo-style ratings, recent form, goals, head-to-head, home-venue
advantage), then uses a LangChain RAG layer with a **local, free Ollama
model** to explain the prediction in plain language using live web-searched
team news. A Flask web UI lets you type a matchup in plain English (e.g.
"Arsenal vs Barcelona, who wins?") and get a point answer.

Runs entirely on a laptop. No paid API required.

## Why it's built this way

The prediction itself is a trained **XGBoost classifier**, not an LLM guess.
Asking a language model to just state a win probability from memory or
vibes produces a plausible-sounding but ungrounded number. Here, the model
computes real probabilities from real features, and the LLM's only jobs are:

1. Estimating structured input features (rough strength, form, goals) from
   retrieved web text — a bounded task with a defined output range
2. Narrating the model's already-computed prediction in natural language

**The LLM never decides which team wins.** That comparison is done in plain
Python (`max(prediction, key=...)`). This was a real bug caught during
development: a small local model, when asked to summarize its own numbers
in prose, misread which probability was actually higher and named the wrong
team as favorite. Moving that decision into code fixed it permanently.

```
                                   ┌─────────────────────────┐
                                   │   data/matches.csv       │
                                   │  (synthetic, 3000 rows)  │
                                   └────────────┬─────────────┘
                                                │
                                                ▼
                                   ┌─────────────────────────┐
                                   │   train_model.py          │
                                   │   XGBoost classifier      │
                                   │   → models/*.joblib       │
                                   └────────────┬─────────────┘
                                                │
   "Arsenal vs Barcelona,          ┌────────────▼─────────────┐
    who wins?"          ─────────▶│   query_predictor.py      │
                                   │  1. parse team names      │
                                   │  2. web_search.py         │
                                   │     (DuckDuckGo + fetch)  │
                                   │  3. Ollama estimates       │
                                   │     input features        │
                                   │  4. XGBoost predicts       │
                                   │     real probabilities     │
                                   │  5. Python picks winner    │
                                   │     (never the LLM)        │
                                   └────────────┬─────────────┘
                                                │
                                                ▼
                                   "Arsenal — 58%"
```

## Stack

| Layer | Tool | Why |
|---|---|---|
| Prediction model | XGBoost | Real probabilities from real features, not LLM guesses |
| Orchestration | LangChain | Ties search → feature extraction → prediction → explanation together |
| Web search | DuckDuckGo (`ddgs`) | Free, no API key |
| Page fetching | `requests` + `BeautifulSoup` | Extracts readable text from search results |
| Local LLM | Ollama (`llama3.2`) | Free, runs on-device, no API costs or rate limits |
| Retrieval | `TFIDFRetriever` (LangChain) | Lightweight keyword retrieval, no embedding model / GPU needed |
| Web UI | Flask | Minimal local server, one form, one result page |

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Ollama (https://ollama.com) and pull a model once:
ollama pull llama3.2

# 1. Generate training data and train the model
python3 data/generate_sample_data.py
python3 src/train_model.py

# 2. Predict a single, fully-specified matchup
python3 src/predict.py

# 3. Ask a natural-language question (live search + local LLM)
python3 src/query_predictor.py "Arsenal vs Barcelona, who wins and what percentage?"

# 4. Or use the web UI
python3 app.py
# then open http://127.0.0.1:5000
```

## Project structure

```
data/generate_sample_data.py   synthetic match dataset (swap for real data later)
src/features.py                shared list of model input columns
src/train_model.py             trains the XGBoost classifier
src/predict.py                 predicts one matchup from known features
src/web_search.py              DuckDuckGo search + page fetch → clean text
src/langchain_agent.py         RAG explanation layer (search context + Ollama)
src/query_predictor.py         full pipeline: NL query → point answer
app.py                         Flask web UI
```

## Current state: synthetic data

The dataset generator creates a realistic but **synthetic** dataset
(≈43% home win / 24% draw / 33% away win, similar to real football) so the
whole pipeline runs immediately without needing a data source. Model
accuracy on this data is ~46%, which is genuinely in the same range as real
bookmaker-grade models — football is hard to predict much above 50-55%
even with rich, real data.

### Known limitation: home-venue feature

Match venue (whether a team is playing at its actual home stadium vs. a
neutral venue like a cup final) is modeled as a binary `true_home_venue`
feature that boosts a team's effective strength when true. It doesn't
encode specific stadiums, cities, or travel effects — just whether the home
advantage applies at all.

## Plugging in real data

Replace the synthetic CSV with real data and use `src/features.py`'s
functions to derive matching columns from raw results. Suggested sources:

- **Historic results:** [football-data.org](https://www.football-data.org),
  [API-Football](https://www.api-football.com)
- **Odds data:** [The Odds API](https://the-odds-api.com) — don't scrape
  betting sites directly; most prohibit it and have anti-bot protection
- **FIFA/game ratings:** Kaggle datasets derived from sofifa.com

## Honest limitations

- **Feature estimation from search text is approximate.** The local LLM
  reads news snippets and estimates numeric signals (form, strength,
  injury impact) — it's a reasonable-effort estimate, not a verified stat.
- **Small local models can misread numbers in prose**, which is why the
  winner is always determined in Python, never asked of the LLM directly.
- **DuckDuckGo's free search has no official API** and can rate-limit under
  heavy use. Fine for development/demo use.
- **This is not betting-grade output.** Treat the point answer as a
  directional estimate for a portfolio project, not a real forecasting
  system — real odds models use far richer, verified structured data.

## License

MIT (or your choice — add a LICENSE file)
