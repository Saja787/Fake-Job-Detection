# 🔎 Fake Job Posting Detection

An end-to-end Machine Learning project for detecting fraudulent job postings using text, categorical, and numerical features.

The project was developed as part of the **Samsung Innovation Campus** training program.

---

## Project Overview

Fake job postings can be used to steal money or personal information from job seekers. They can also waste applicants' time and reduce trust in online job platforms.

The goal of this project is to build a Machine Learning system that classifies a job posting as:

- **Real (0)**
- **Fraudulent (1)**

The project covers the complete Machine Learning workflow:

**Data Cleaning → Feature Engineering → EDA → Model Training → Evaluation → Model Comparison → Ensemble Learning → Deployment**

---

##  Problem Statement

This project is formulated as a **supervised binary classification problem** with a highly imbalanced target.

The model analyzes job posting text and metadata to identify potentially fraudulent listings.

---

##  Dataset

The project uses the **EMSCAD Fake Job Postings dataset**.

- **17,880 job postings**
- **18 original attributes**
- **17,014 Real postings**
- **866 Fraudulent postings**
- **4.8% fraudulent postings**
- 80/20 stratified train-test split
- `random_state = 42`

### Main Features

**Text Features:**
- `title`
- `location`
- `department`
- `company_profile`
- `description`
- `requirements`
- `benefits`

**Categorical Features:**
- `employment_type`
- `required_experience`
- `required_education`
- `industry`
- `function`

**Binary Features:**
- `telecommuting`
- `has_company_logo`
- `has_questions`

---

## Data Cleaning & Preprocessing

The preprocessing pipeline included:

- Handling missing values
- Duplicate detection
- Text cleaning
- Lowercasing
- HTML removal
- URL removal
- Email removal
- Punctuation removal
- Whitespace normalization
- Categorical encoding
- Numerical scaling
- TF-IDF feature extraction
- Class imbalance handling

No rows were dropped during the cleaning process, and no fully identical duplicate rows were detected.

---

##  Feature Engineering

Additional features were created to provide structural and information-availability signals.

| Feature | Description |
|---|---|
| `missing_fields_count` | Number of missing fields in a job posting |
| `has_salary` | Whether salary information is available |
| `has_description` | Whether a description is available |
| `has_company_profile` | Whether a company profile is available |
| `has_requirements` | Whether requirements are available |
| `has_benefits` | Whether benefits are available |
| `has_location` | Whether location information is available |
| `job_context` | Combined title + industry + function |
| `title_len` | Length of the job title |
| `description_len` | Length of the job description |

---

# Experiments

Seven experiments were conducted to compare different feature representations, encoding methods, imbalance-handling techniques, ensemble methods, and transformer-based features.

## EX1 — One-Hot Encoding + Numerical Features

Used categorical and numerical features only.

- One-Hot Encoding for categorical features
- StandardScaler for numerical features
- Six Machine Learning models were evaluated

**Best Model:** Random Forest

**F1-Score:** 0.754

---

## EX2 — Ordinal Encoding + Numerical Features

Used the same categorical and numerical features as EX1, but replaced One-Hot Encoding with Ordinal Encoding.

**Best Model:** Random Forest

**F1-Score:** 0.728

One-Hot Encoding performed better than Ordinal Encoding, so One-Hot Encoding was preferred in later experiments.

---

## EX3 — Classical ML + TF-IDF

Introduced text information using TF-IDF features from `job_context`, combined with categorical and numerical features.

**Best Model:** Logistic Regression

**F1-Score:** 0.720

---

## EX4 — Text Context + Numerical Features

A richer text representation was created by combining:

- `title`
- `description`
- `requirements`
- `benefits`

The combined text was transformed using TF-IDF and paired with numerical and information-availability features.

**Best Model:** Logistic Regression

**F1-Score:** 0.822

This showed a significant improvement compared with EX3.

---

## EX5 — Imbalance Handling

Because fraudulent postings represented only **4.8%** of the dataset, imbalance-handling techniques were evaluated.

The following approaches were tested:

- Random Over-Sampling (ROS)
- SMOTE

These techniques were applied to the training data only.

**Best Model:** Linear SVC + SMOTE

**F1-Score:** 0.826

---

## EX6 — Ensemble Learning

The strongest models from previous experiments were combined using ensemble methods.

### Hard Voting

Uses the majority decision of multiple models.

### Stacking

Uses the predictions of multiple base models as inputs to a Logistic Regression meta-model that learns how to combine their predictions.

The base models included:

- Logistic Regression + ROS
- Linear SVC + ROS
- Random Forest + SMOTE
- XGBoost + SMOTE

**Best Model:** Stacking Ensemble

**F1-Score:** 0.850

---

## EX7 — DistilBERT Feature Extraction

A pre-trained **DistilBERT** model was used as a frozen feature extractor.

Pipeline:

**Job Text → Tokenization → Frozen DistilBERT → Mean-Pooled Embeddings → Numerical Features → XGBoost**

The best configuration used a tuned decision threshold of **0.22**.

**Best Model:** DistilBERT + XGBoost

**F1-Score:** 0.773

DistilBERT provided strong results but did not outperform the Stacking Ensemble.

---

# Overall Results

The following table summarizes the best model from each experiment.

| Experiment | Best Model | F1-Score | Precision | Recall | ROC-AUC | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| EX1 | Random Forest | 0.754 | 0.964 | 0.618 | 0.956 | ≈ 0.981 |
| EX2 | Random Forest | 0.728 | 0.936 | 0.595 | 0.950 | ≈ 0.977 |
| EX3 | Logistic Regression | 0.720 | 0.706 | 0.734 | 0.946 | ≈ 0.972 |
| EX4 | Logistic Regression | 0.822 | 0.842 | 0.803 | 0.983 | ≈ 0.983 |
| EX5 | Linear SVC + SMOTE | 0.826 | 0.893 | 0.769 | 0.980 | ≈ 0.984 |
| **EX6** | **Stacking** | **0.850** | **0.950** | **0.769** | **0.983** | **0.987** |
| EX7 | DistilBERT + XGBoost | 0.773 | 0.881 | 0.688 | 0.982 | 0.980 |

###  Main Findings

- **EX1:** Random Forest performed strongly using only structured features.
- **EX2:** One-Hot Encoding performed better than Ordinal Encoding.
- **EX3:** Adding limited text information did not improve F1.
- **EX4:** Using richer text features significantly improved performance.
- **EX5:** SMOTE improved the balance between Precision and Recall.
- **EX6:** Stacking achieved the best overall performance.
- **EX7:** DistilBERT produced strong results but did not outperform the ensemble approach.

---

# Final Model

The **Stacking Ensemble from EX6** was selected as the final model because it achieved the highest F1-score and provided the best overall balance between Precision and Recall.

### Final Performance

| Metric | Score |
|---|---:|
| **Accuracy** | **98.7%** |
| **Precision** | **95.0%** |
| **Recall** | **76.9%** |
| **F1-Score** | **85.0%** |
| **ROC-AUC** | **98.3%** |

The final Stacking model was saved using **Joblib** and integrated into the Streamlit deployment.

---

#  Deployment

The trained model was deployed using **Streamlit**.
Live Demo:

https://fake-job-detector-bgkwdduaa86fvablkapgh7.streamlit.app/

The deployment folder contains:

```text
deployment/
│
├── app.py
├── feature_engineering.py
├── train_model.py
├── requirements.txt
└── fake_job_stacking_model.joblib
