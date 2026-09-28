import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import FEATURE_COLUMNS

LABEL_MAP = {"H": 0, "D": 1, "A": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_data(path="data/matches.csv"):
    df = pd.read_csv(path)
    df["label"] = df["result"].map(LABEL_MAP)
    return df


def train():
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("Log loss:", round(log_loss(y_test, probs), 4))
    print(classification_report(y_test, preds, target_names=["Home", "Draw", "Away"]))

    importances = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: -x[1]
    )
    print("\nFeature importances:")
    for name, imp in importances:
        print(f"  {name:20s} {imp:.3f}")

    joblib.dump(model, "models/outcome_model.joblib")
    print("\nSaved model to models/outcome_model.joblib")


if __name__ == "__main__":
    train()
