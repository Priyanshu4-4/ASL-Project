"""
Train a Random Forest classifier on collected hand-landmark data
(data/landmarks.csv) and save it to model.joblib.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

CSV_PATH = "data/landmarks.csv"
MODEL_PATH = "model.joblib"


def main():
    df = pd.read_csv(CSV_PATH)
    if len(df) < 50:
        print(
            f"Only {len(df)} samples found — collect more data with "
            "collect_data.py before training (aim for 100+ per letter)."
        )
        return

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds))

    joblib.dump(clf, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
