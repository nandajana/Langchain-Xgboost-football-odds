"""
Simple local web UI for the football predictor.

Run: python3 app.py
Then open: http://127.0.0.1:5000 in your browser.
"""
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, render_template_string, request

from query_predictor import predict_from_query

app = Flask(__name__)

PAGE = """
<!doctype html>
<html>
<head>
  <title>Football Match Predictor</title>
  <style>
    body { font-family: sans-serif; max-width: 640px; margin: 40px auto; padding: 0 20px; }
    h1 { font-size: 22px; }
    input[type=text] { width: 100%; padding: 10px; font-size: 16px; box-sizing: border-box; }
    button { margin-top: 10px; padding: 10px 20px; font-size: 16px; cursor: pointer; }
    .result { margin-top: 30px; padding: 20px; background: #f4f4f4; border-radius: 8px; }
    .point-answer { font-size: 24px; font-weight: bold; margin-bottom: 10px; }
    .probs { margin: 10px 0; }
    .probs div { margin: 4px 0; }
    .error { color: #b00020; }
    .loading { color: #666; font-style: italic; }
    .signals { margin-top: 15px; font-size: 13px; color: #555; }
  </style>
</head>
<body>
  <h1>Football Match Predictor</h1>
  <form method="post" onsubmit="document.getElementById('loading').style.display='block';">
    <input type="text" name="query" placeholder="e.g. Arsenal vs Barcelona, who wins?"
           value="{{ query or '' }}" required>
    <button type="submit">Predict</button>
  </form>
  <p id="loading" class="loading" style="display:none;">
    Searching and predicting... this can take a minute or two.
  </p>

  {% if error %}
    <div class="result error"><strong>Error:</strong> {{ error }}</div>
  {% endif %}

  {% if result %}
    <div class="result">
      <div class="point-answer">{{ result.point_answer }}</div>
      <div class="probs">
        {% for outcome, vals in result.prediction.items() %}
          <div>{{ outcome }}: {{ "%.1f"|format(vals.probability * 100) }}%
               (fair odds {{ vals.fair_decimal_odds }})</div>
        {% endfor %}
      </div>
      <div class="signals">
        {{ result.home_team }} signals: {{ result.home_signals }}<br>
        {{ result.away_team }} signals: {{ result.away_signals }}
      </div>
    </div>
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    query = None
    result = None
    error = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            try:
                out = predict_from_query(query)
                if "error" in out:
                    error = out["error"]
                else:
                    result = out
            except Exception:
                error = "Something went wrong running the prediction. Check the terminal for details."
                traceback.print_exc()

    return render_template_string(PAGE, query=query, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
