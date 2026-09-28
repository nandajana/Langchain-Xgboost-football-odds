"""
LangChain layer on top of the trained model.

Important design choice: the LLM never invents the odds. The XGBoost model
(train_model.py / predict.py) computes the actual probabilities. This module
only RETRIEVES supporting context (team news via web_search.py) and asks the
LLM to EXPLAIN the already-computed numbers in natural language.

Uses a local Ollama model (free, runs on your laptop, no API key). Make sure
Ollama is running and you've pulled a model: ollama pull llama3.2
"""
import os

from langchain_community.retrievers import TFIDFRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from predict import predict_match
from web_search import search_and_build_documents

# ---------------------------------------------------------------------------
# 1. Static fallback notes (used if live search fails or you want offline demo)
# ---------------------------------------------------------------------------
SAMPLE_DOCS = [
    Document(
        page_content="Arsenal have won 4 of their last 5 home matches, with "
                      "their attack averaging 2.3 goals per game since the "
                      "international break.",
        metadata={"team": "Arsenal", "type": "form"},
    ),
    Document(
        page_content="Man City's key midfielder is a doubt for the upcoming "
                      "fixture due to a hamstring issue picked up in training.",
        metadata={"team": "Man City", "type": "injury"},
    ),
]


def build_retriever(docs=None, k=3):
    docs = docs or SAMPLE_DOCS
    return TFIDFRetriever.from_documents(docs, k=min(k, len(docs)))


def build_live_retriever(home_team, away_team, k=4):
    """Search the web for this matchup and build a retriever from real pages."""
    query = f"{home_team} vs {away_team} team news injuries form"
    try:
        raw_docs = search_and_build_documents(query, max_results=k)
    except Exception as e:
        print(f"[live search failed: {e}] falling back to local notes")
        raw_docs = []

    if not raw_docs:
        return build_retriever(SAMPLE_DOCS)

    docs = [Document(page_content=d["text"], metadata={"source": d["source"]})
            for d in raw_docs]
    return build_retriever(docs, k=k)


# ---------------------------------------------------------------------------
# 2. RAG explanation chain
# ---------------------------------------------------------------------------
EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a football analyst. You are given MODEL-COMPUTED probabilities "
     "(already calculated, not yours to change) and some retrieved context "
     "notes. Explain the prediction in 3-4 sentences, referencing the "
     "context where relevant. Never state a probability or odds figure that "
     "isn't in the MODEL OUTPUT provided to you."),
    ("human",
     "Match: {home_team} vs {away_team}\n\n"
     "MODEL OUTPUT (ground truth, do not alter):\n{model_output}\n\n"
     "RETRIEVED CONTEXT:\n{context}\n\n"
     "Question: {question}"),
])


def explain_prediction(home_team, away_team, match_features,
                        question="Why is this the predicted outcome?",
                        model_path="models/outcome_model.joblib",
                        use_live_search=False):
    prediction = predict_match(model_path, match_features)
    model_output_str = "\n".join(
        f"  {outcome}: probability={vals['probability']:.1%}, "
        f"fair decimal odds={vals['fair_decimal_odds']}"
        for outcome, vals in prediction.items()
    )

    retriever = build_live_retriever(home_team, away_team) if use_live_search else build_retriever()
    retrieved = retriever.invoke(f"{home_team} {away_team} {question}")
    context_str = "\n".join(
        f"- {d.page_content[:400]} (source: {d.metadata.get('source', 'local notes')})"
        for d in retrieved
    ) or "(no relevant notes found)"

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=os.environ.get("OLLAMA_MODEL", "llama3.2"), temperature=0.2)
    chain = EXPLAIN_PROMPT | llm
    try:
        result = chain.invoke({
            "home_team": home_team,
            "away_team": away_team,
            "model_output": model_output_str,
            "context": context_str,
            "question": question,
        })
        explanation = result.content
    except Exception as e:
        explanation = (f"[Ollama call failed: {e}. Is Ollama running and "
                       f"the model pulled? Try: ollama list]")

    return {
        "prediction": prediction,
        "context_used": context_str,
        "explanation": explanation,
    }


if __name__ == "__main__":
    example_features = {
        "home_elo": 1620, "away_elo": 1550,
        "home_fifa_rating": 82, "away_fifa_rating": 78,
        "home_form_pts": 11, "away_form_pts": 7,
        "home_goals_avg": 2.1, "away_goals_avg": 1.4,
        "h2h_home_win_rate": 0.6,
        "true_home_venue": 1,
    }
    out = explain_prediction("Arsenal", "Man City", example_features,
                              use_live_search=True)
    print("PREDICTION:", out["prediction"])
    print("\nCONTEXT USED:\n", out["context_used"])
    print("\nEXPLANATION:\n", out["explanation"])
