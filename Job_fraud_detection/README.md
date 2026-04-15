# 🔍 Job Fraud Detection System

> Detect fake job postings using NLP and Deep Learning (LSTM, CNN, Random Forest, Logistic Regression)

---

## 📌 Overview

Fake job postings are a growing threat — this project builds an **end-to-end ML system** to classify job listings as **Fraudulent** or **Legitimate** using multiple machine learning and deep learning models.

Built with Django as the web interface, the system lets users:
- View dataset statistics
- Compare 4 ML model performances
- Predict fraud risk for any job posting in real time

---

## 🧠 Models Used

| Model | Type | Key Strength |
|---|---|---|
| Logistic Regression | Classical ML | Fast, interpretable baseline (TF-IDF) |
| Random Forest | Ensemble | Handles non-linear patterns |
| CNN (1D) | Deep Learning | Feature extraction from text sequences |
| LSTM (Bidirectional) | Deep Learning | Captures long-range text dependencies |

---

## 🗂️ Project Structure

```
Job_fraud_detection/
├── README.md
├── requirements.txt
├── model_training.py        # Train all 4 models
└── predict.py               # Standalone prediction script
```

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Web Framework**: Django 6.0
- **ML Libraries**: scikit-learn, TensorFlow/Keras
- **NLP**: TF-IDF Vectorizer, Keras Tokenizer
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Database**: SQLite

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/pyverse/PyVerse.git
cd PyVerse/Job_fraud_detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run migrations
python manage.py migrate

# 5. Start the server
python manage.py runserver
```

Visit: `http://localhost:8000`

---

## 📊 Dataset

- **Source**: [EMSCAD - Employment Scam Aegean Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Size**: ~17,880 job postings
- **Split**: 70% Train / 15% Validation / 15% Test
- **Class balance**: ~95% Legitimate, ~5% Fraudulent (imbalanced)

**Expected CSV columns:**
```
title, description, location, label
```
Where `label` = 0 (Legitimate) or 1 (Fraudulent)

---

## 🚀 Quick Prediction (Without Django)

```python
from predict import predict_fraud

result = predict_fraud(
    title="Data Scientist",
    description="Work from home, earn $5000/week, no experience needed!",
    location="Remote"
)

print(result)
# Output: {'label': 'Fraudulent', 'confidence': 0.94, 'risk_level': 'High'}
```

---

## 🏋️ Training the Models

```bash
python model_training.py
```

This will:
1. Load `ml/data/jobs_dataset.csv`
2. Preprocess text with TF-IDF
3. Train all 4 models
4. Save models to `ml/models/`
5. Save metrics to `ml/metrics/`
6. Generate confusion matrix and ROC plots

---

## 📈 Model Performance (Sample Results)

| Model | Accuracy | F1 Score | AUC |
|---|---|---|---|
| Logistic Regression | 97.8% | 0.81 | 0.96 |
| Random Forest | 98.2% | 0.84 | 0.97 |
| CNN | 98.5% | 0.86 | 0.98 |
| LSTM (BiLSTM) | 98.9% | 0.88 | 0.99 |

---

## 🖥️ App Pages

| Page | URL | Description |
|---|---|---|
| Dashboard | `/` | Quick navigation to all sections |
| Dataset | `/dataset/` | Dataset stats & distribution |
| Algorithms | `/algorithms/` | Per-model metrics, confusion matrix, ROC |
| Comparison | `/comparison/` | Side-by-side model performance |
| Prediction | `/prediction/` | Real-time fraud prediction form |

---

## 📸 Screenshots

Screenshots are not currently included in this repository snapshot. Add the referenced images to a committed `screenshots/` directory or update this section with valid image URLs before publishing.
---

## 🔮 Future Enhancements

- SHAP/LIME for model explainability
- REST API endpoints for programmatic access
- Docker deployment support
- Real-time model performance dashboard
- User authentication & prediction history

---

## 👩‍💻 Author

**Srinithi** — AI & Data Science Student, Jerusalem College of Engineering  
📧 [LinkedIn](https://www.linkedin.com/in/srinithi-maran558)  
🐙 [GitHub](https://github.com/srinithithirumaran)

---

## 📄 License

This project is open-source and available under the [MIT License](../LICENSE).
