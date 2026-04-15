"""
predict.py
----------
Standalone fraud prediction script for Job Fraud Detection System.
Loads pre-trained Logistic Regression model and predicts fraud probability.

Usage:
    python predict.py
    or import as module: from predict import predict_fraud
"""

import joblib
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "ml", "models", "logistic_regression.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "ml", "models", "vectorizer.pkl")


def load_model():
    """Load the saved model and vectorizer from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Model files not found. Please run model_training.py first.\n"
            f"Expected: {MODEL_PATH} and {VECTORIZER_PATH}"
        )
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_fraud(title: str, description: str, location: str) -> dict:
    """
    Predict whether a job posting is fraudulent.

    Parameters
    ----------
    title       : str  — Job title
    description : str  — Job description text
    location    : str  — Job location

    Returns
    -------
    dict with keys:
        'label'      : 'Fraudulent' or 'Legitimate'
        'confidence' : float (0.0 – 1.0), probability of fraud
        'risk_level' : 'High' | 'Medium' | 'Low'
    """
    model, vectorizer = load_model()

    # Combine all text fields
    combined_text = f"{title} {description} {location}".lower().strip()

    # Vectorize
    X = vectorizer.transform([combined_text])

    # Predict
    prob  = model.predict_proba(X)[0, 1]   # probability of class 1 (fraud)
    label = "Fraudulent" if prob > 0.5 else "Legitimate"

    # Risk level
    if prob >= 0.75:
        risk = "High"
    elif prob >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "label":      label,
        "confidence": round(float(prob), 4),
        "risk_level": risk,
    }


# ─── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Sample 1 — Suspicious posting
    sample_1 = {
        "title":       "Work From Home – Earn $5000/week",
        "description": "No experience needed! Guaranteed income. Send personal details to claim your offer now.",
        "location":    "Remote – Worldwide",
    }

    # Sample 2 — Legitimate posting
    sample_2 = {
        "title":       "Data Scientist",
        "description": "We are looking for an experienced data scientist to join our analytics team. "
                       "Responsibilities include building ML models, data pipelines, and dashboards.",
        "location":    "Bangalore, India",
    }

    print("=" * 55)
    print("       JOB FRAUD DETECTION — PREDICTION DEMO")
    print("=" * 55)

    for i, sample in enumerate([sample_1, sample_2], 1):
        try:
            result = predict_fraud(**sample)
            print(f"\nSample {i}: {sample['title']}")
            print(f"  Label      : {result['label']}")
            print(f"  Confidence : {result['confidence'] * 100:.1f}%")
            print(f"  Risk Level : {result['risk_level']}")
        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
            break

    print("\n" + "=" * 55)
