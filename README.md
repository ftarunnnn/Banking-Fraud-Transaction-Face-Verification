# 🏦 Banking — Fraud Transaction & Face Verification System

An end-to-end Machine Learning and Deep Learning solution designed for modern banking applications. This system combines **ML-based Fraud Detection** (Random Forest / XGBoost) with **DL-based Biometric Face Verification** (PyTorch Facial Embedding ResNet) to secure digital transactions against fraud.

---

## 📌 Project Overview & Architecture

```
                                  [ Incoming Transaction Payload ]
                                                 │
                                                 ▼
                                    ┌────────────────────────┐
                                    │  Phase 5 & 6: ML Model │
                                    │  Fraud Detection       │
                                    └────────────┬───────────┘
                                                 │
                                  ┌──────────────┴──────────────┐
                                  ▼                             ▼
                          [ Low Risk (<30%) ]          [ High Risk (>=30%) ]
                                  │                             │
                                  ▼                             ▼
                         🟢 Transaction           🔒 Biometric Challenge
                            APPROVED              ┌─────────────┴────────────┐
                                                  │ Phase 7 & 8: Deep Learning│
                                                  │ Face Verification        │
                                                  └─────────────┬────────────┘
                                                                │
                                                 ┌──────────────┴─────────────┐
                                                 ▼                            ▼
                                        👤 Face Verified           ❌ Verification Failed
                                                 │                            │
                                                 ▼                            ▼
                                        🟢 Transaction               🚨 Transaction
                                           APPROVED                     BLOCKED
```

---

## 🚀 10 Project Phases

- [x] **Phase 1 — Requirement & Dataset Collection**: Generation and structuring of banking transaction datasets and user face datasets.
- [x] **Phase 2 — Data Preprocessing**: Handling missing/duplicate values, encoding categorical features, resizing and normalizing face images.
- [x] **Phase 3 — Exploratory Data Analysis (EDA)**: Comprehensive visualization of fraud vs. normal transaction patterns, transaction amount distributions, and class imbalance.
- [x] **Phase 4 — Transaction Feature Engineering**: Calculating transaction frequency, rolling averages, location/device anomaly flags, and time-based features.
- [x] **Phase 5 — ML Fraud Detection**: Training Random Forest and XGBoost classifiers with class weighting for imbalanced data.
- [x] **Phase 6 — ML Model Evaluation**: Metric reporting (Accuracy, Precision, Recall, F1, ROC-AUC) and visualization (Confusion Matrix, ROC Curve, PR Curve).
- [x] **Phase 7 — Face Image Preprocessing**: Face detection, alignment, cropping, and standard pixel normalization.
- [x] **Phase 8 — DL Face Verification**: PyTorch Deep Learning Face Embedding Network and Cosine Similarity identity matcher.
- [x] **Phase 9 — Integrated Banking System**: End-to-end transaction fraud check and biometric face verification pipeline.
- [x] **Phase 10 — Deployment & Web Application**: Interactive Streamlit dashboard with real-time transaction simulator, biometric camera/upload verification, and analytics.

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ftarunnnn/Banking-Fraud-Transaction-Face-Verification.git
   cd Banking-Fraud-Transaction-Face-Verification
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Full Pipeline**:
   ```bash
   python src/banking_pipeline.py
   ```

4. **Launch Streamlit Web App**:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Results Summary

- **Fraud Detection ML (XGBoost / Random Forest)**: High ROC-AUC score on imbalanced banking transactions.
- **Face Verification DL Model**: High identity verification accuracy based on deep face embedding cosine distance.

---
*Created by [ftarunnnn](https://github.com/ftarunnnn)*
