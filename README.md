# Fake Job Posting Detection

An end-to-end machine learning project for detecting fraudulent job postings using text, categorical, and numerical features.

## Project Overview

Fake job postings can be used to steal money or personal information from job seekers.

The goal of this project is to build a machine learning system that classifies job postings as:

- Real
- Fraudulent

The project uses the EMSCAD fake job postings dataset containing 17,880 job postings, with approximately 4.8% fraudulent postings.

## Dataset

- 17,880 job postings
- 18 original attributes
- 4.8% fraudulent postings
- Text, categorical, and binary features

Main text features include:

- title
- company_profile
- description
- requirements
- benefits

## Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Duplicate detection
- Text cleaning
- HTML removal
- URL removal
- Email removal
- Punctuation removal
- Feature engineering
- Text combination
- TF-IDF
- Categorical encoding
- Numerical scaling
- Class imbalance handling

## Feature Engineering

Additional features include:

- missing_fields_count
- has_salary
- has_description
- has_company_profile
- has_requirements
- has_benefits
- has_location
- job_context
- title_len
- description_len

## Experiments

Seven experiments were conducted:

1. One-Hot Encoding + Numerical Features
2. Ordinal Encoding + Numerical Features
3. Classical ML + TF-IDF
4. Text Context + Numerical Features
5. Imbalance Handling
6. Ensemble Learning
7. DistilBERT Feature Extraction

## Best Model

The best-performing model was the Stacking Ensemble.

| Metric | Score |
|---|---:|
| Accuracy | 0.987 |
| Precision | 0.950 |
| Recall | 0.769 |
| F1-Score | 0.850 |
| ROC-AUC | 0.983 |

The Stacking model provided the best overall balance between precision and recall.

## Deployment

The model was deployed using Streamlit.

Live Demo:

https://fake-job-detector-bgkwdduaa86fvablkapgh7.streamlit.app/

## Project Files

- `notebooks/` - Main machine learning notebook
- `deployment/` - Streamlit application and trained model
- `data/` - Dataset
- `reports/` - Final project report
- `presentation/` - Project presentation

## Project Resources

GitHub Repository:
https://github.com/Mohamed934789/fake-job-detector

Live Demo:
https://fake-job-detector-bgkwdduaa86fvablkapgh7.streamlit.app/

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TF-IDF
- SMOTE
- DistilBERT
- Streamlit
- Joblib

## Team

Samsung Innovation Campus

- Mohamed Kassab
- Hamdy Osama
- Yousef Mohamed
- Saja Elsayed
