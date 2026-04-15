"""
model_training.py
-----------------
Train all 4 ML models for Job Fraud Detection:
  1. Logistic Regression  (TF-IDF baseline)
  2. Random Forest        (Ensemble)
  3. CNN                  (1D Convolutional Neural Network)
  4. BiLSTM              (Bidirectional LSTM)

Run:
    python model_training.py

Outputs saved to:
    ml/models/   — .pkl and .h5 model files
    ml/metrics/  — JSON metric files
    ml/plots/    — Confusion matrix and ROC curve images
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, "ml", "data", "jobs_dataset.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "ml", "models")
METRICS_DIR  = os.path.join(BASE_DIR, "ml", "metrics")
PLOTS_DIR    = os.path.join(BASE_DIR, "ml", "plots")

for d in [MODELS_DIR, METRICS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ─── 1. Load & Preprocess Data ────────────────────────────────────────────────
def load_data(path: str):
    """Load CSV and return combined text + labels."""
    print(f"[INFO] Loading dataset from {path} ...")
    df = pd.read_csv(path)

    # Fill missing values
    for col in ["title", "description", "location"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    df["text"]  = df["title"] + " " + df["description"] + " " + df["location"]
    df["text"]  = df["text"].str.lower().str.strip()
    df["label"] = df["label"].astype(int)

    print(f"[INFO] Total samples : {len(df)}")
    print(f"[INFO] Fraudulent    : {df['label'].sum()}")
    print(f"[INFO] Legitimate    : {(df['label'] == 0).sum()}")
    return df["text"].values, df["label"].values


# ─── 2. Metrics Helper ────────────────────────────────────────────────────────
def compute_metrics(name, y_true, y_pred, y_prob):
    metrics = {
        "model":    name,
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1":       round(f1_score(y_true, y_pred), 4),
        "auc":      round(roc_auc_score(y_true, y_prob), 4),
    }
    print(f"\n[RESULT] {name}")
    print(f"  Accuracy : {metrics['accuracy']}%")
    print(f"  F1 Score : {metrics['f1']}")
    print(f"  AUC      : {metrics['auc']}")
    return metrics


def save_metrics(name, metrics):
    path = os.path.join(METRICS_DIR, f"{name.lower().replace(' ', '_')}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVED] Metrics → {path}")


def plot_confusion_matrix(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Fraudulent"],
                yticklabels=["Legitimate", "Fraudulent"])
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"cm_{name.lower().replace(' ', '_')}.png")
    plt.savefig(path)
    plt.close()
    print(f"[SAVED] Confusion Matrix → {path}")


def plot_roc_curve(name, y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"roc_{name.lower().replace(' ', '_')}.png")
    plt.savefig(path)
    plt.close()
    print(f"[SAVED] ROC Curve → {path}")


# ─── 3. Classical ML Models ───────────────────────────────────────────────────
def train_classical_models(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print("  TRAINING CLASSICAL ML MODELS")
    print("=" * 50)

    # TF-IDF Vectorizer
    print("\n[INFO] Fitting TF-IDF Vectorizer ...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))
    print("[SAVED] Vectorizer → ml/models/vectorizer.pkl")

    results = []

    # --- Logistic Regression ---
    print("\n[TRAINING] Logistic Regression ...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_tfidf, y_train)
    y_pred = lr.predict(X_test_tfidf)
    y_prob = lr.predict_proba(X_test_tfidf)[:, 1]

    m = compute_metrics("Logistic Regression", y_test, y_pred, y_prob)
    save_metrics("logistic_regression", m)
    plot_confusion_matrix("Logistic Regression", y_test, y_pred)
    plot_roc_curve("Logistic Regression", y_test, y_prob)
    joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    results.append(m)

    # --- Random Forest ---
    print("\n[TRAINING] Random Forest ...")
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train_tfidf, y_train)
    y_pred = rf.predict(X_test_tfidf)
    y_prob = rf.predict_proba(X_test_tfidf)[:, 1]

    m = compute_metrics("Random Forest", y_test, y_pred, y_prob)
    save_metrics("random_forest", m)
    plot_confusion_matrix("Random Forest", y_test, y_pred)
    plot_roc_curve("Random Forest", y_test, y_prob)
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    results.append(m)

    return results


# ─── 4. Deep Learning Models ──────────────────────────────────────────────────
def train_deep_learning_models(X_train, X_test, y_train, y_test):
    try:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Embedding, Conv1D, GlobalMaxPooling1D,
            LSTM, Bidirectional, Dense, Dropout
        )
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        print("\n[SKIP] TensorFlow not found. Skipping CNN and LSTM training.")
        print("       Install with: pip install tensorflow")
        return []

    print("\n" + "=" * 50)
    print("  TRAINING DEEP LEARNING MODELS")
    print("=" * 50)

    MAX_WORDS = 15000
    MAX_LEN   = 200

    print("\n[INFO] Tokenizing text ...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
    X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=MAX_LEN)

    joblib.dump(tokenizer, os.path.join(MODELS_DIR, "keras_tokenizer.pkl"))

    results  = []
    es       = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # --- CNN ---
    print("\n[TRAINING] CNN (1D Convolutional) ...")
    cnn = Sequential([
        Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
        Conv1D(128, 5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    cnn.fit(X_train_seq, y_train,
            validation_split=0.1, epochs=10,
            batch_size=64, callbacks=[es], verbose=1)

    y_prob = cnn.predict(X_test_seq).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    m = compute_metrics("CNN", y_test, y_pred, y_prob)
    save_metrics("cnn", m)
    plot_confusion_matrix("CNN", y_test, y_pred)
    plot_roc_curve("CNN", y_test, y_prob)
    cnn.save(os.path.join(MODELS_DIR, "cnn_model.h5"))
    results.append(m)

    # --- BiLSTM ---
    print("\n[TRAINING] Bidirectional LSTM ...")
    lstm = Sequential([
        Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    lstm.fit(X_train_seq, y_train,
             validation_split=0.1, epochs=10,
             batch_size=64, callbacks=[es], verbose=1)

    y_prob = lstm.predict(X_test_seq).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    m = compute_metrics("BiLSTM", y_test, y_pred, y_prob)
    save_metrics("bilstm", m)
    plot_confusion_matrix("BiLSTM", y_test, y_pred)
    plot_roc_curve("BiLSTM", y_test, y_prob)
    lstm.save(os.path.join(MODELS_DIR, "lstm_model.h5"))
    results.append(m)

    return results


# ─── 5. Summary ───────────────────────────────────────────────────────────────
def print_summary(all_results):
    print("\n" + "=" * 55)
    print("              FINAL MODEL COMPARISON")
    print("=" * 55)
    print(f"{'Model':<22} {'Accuracy':>10} {'F1':>8} {'AUC':>8}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['model']:<22} {r['accuracy']:>9}% {r['f1']:>8} {r['auc']:>8}")
    print("=" * 55)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at: {DATA_PATH}")
        print("        Download from: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction")
        print("        Rename to 'jobs_dataset.csv' and place in ml/data/")
        exit(1)

    X, y = load_data(DATA_PATH)

    # Train / test split (validation is handled internally where needed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"\n[INFO] Train : {len(X_train)} | Test : {len(X_test)}")

    classical_results = train_classical_models(X_train, X_test, y_train, y_test)
    dl_results        = train_deep_learning_models(X_train, X_test, y_train, y_test)

    print_summary(classical_results + dl_results)
    print("\n✅ Training complete! All models saved to ml/models/")
