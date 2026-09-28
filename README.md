# Football Match Outcome Predictor — XGBoost + LangChain RAG

Predicts Win/Draw/Loss probabilities for a football match from multi-factor
team data (Elo-style ratings, recent form, goals, head-to-head, home-venue
advantage), then uses a LangChain RAG layer with a **local, free Ollama
model** to explain the prediction in plain language using live web-searched
team news. A Flask web UI lets you type a matchup in plain English (e.g.
"Arsenal vs Barcelona, who wins?") and get a point answer.

Runs entirely on a laptop. No paid API required.


<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/aced3ad5-e3be-4534-8377-5822f9012c28" />


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

## Setup

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

## Structure

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





## Data Useage
Replace the synthetic CSV with real data and use `src/features.py`'s
functions to derive matching columns from raw results. Suggested sources:

- **Historic results:** [football-data.org](https://www.football-data.org),
  [API-Football](https://www.api-football.com)
- **Odds data:** [The Odds API](https://the-odds-api.com) — don't scrape
  betting sites directly; most prohibit it and have anti-bot protection
- **FIFA/game ratings:** Kaggle datasets derived from sofifa.com

