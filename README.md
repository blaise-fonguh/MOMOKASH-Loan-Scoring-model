MOMOKASH Behavioural Loan Scoring Engine

A production-grade, unsupervised behavioural credit scoring system developed for MOMOKASH, a micro-lending platform serving 20,000+ users across Cameroon.
The system automates credit-limit assignment using historical behavioural data and machine learning, enabling consistent, fair, and scalable lending decisions.

🚀 Project Overview

Traditional manual credit review slowed MOMOKASH’s loan approvals and exposed the platform to inconsistent decisions.
This project solves that by:

Building an end-to-end data integration pipeline (loans, refunds, penalties, debts – 3 years of history).

Engineering a behavioural feature layer capturing real repayment behaviour.

Training an unsupervised K-Means model with automated K-selection.

Packaging the entire system into a deployable .pkl scoring engine.

The model clusters borrowers into risk tiers and maps them into credit-limit bands (500–10,000 FCFA).

📊 Key Features
1. Data Engineering & Cleaning

Consolidated multi-source data into a unified integrated_data.csv.

Enforced a strict modelling window to avoid leakage (Sept 2022 — Sept 2025).

Cleaned missing values, inconsistencies, and abnormal borrower histories.

2. Feature Engineering

Constructed a behavioural feature layer including:

Repayment rate

Debt-to-limit ratio

Borrowing frequency

Penalty patterns

Refund consistency

Tenure and usage patterns

3. Unsupervised Risk Modelling

Trained a K-Means clustering model.

Evaluated clusters using Silhouette, Davies–Bouldin, and Calinski–Harabasz.

📌 Final Silhouette Score: 0.376
(Indicates strong behavioural separation for risk segmentation.)

4. Credit-Limit Mapping

Risk clusters are mapped to limit bands:

Cluster	Risk Level	Assigned Limit
0	High	500–2,000 FCFA
1	Medium	3,000–5,000 FCFA
2	Low	6,000–10,000 FCFA
5. Production Scoring Engine

Packaged as:

scoring_engine.pkl


Includes:

StandardScaler preprocessing

K-Means model

Mapping dictionary

Predict → Assign Limit → Export

Deployable as:

A microservice

An event-driven API

An internal scoring function for MOMOKASH backend

🧱 Tech Stack

Python

Pandas

Scikit-learn

NumPy

Jupyter Notebook

Pickle (.pkl) model packaging

📌 Repository Structure
├── data/
│   ├── integrated_data.csv
│   └── raw/ 
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_kmeans_model.ipynb
│   └── 04_scoring_engine.ipynb
│
├── models/
│   └── scoring_engine.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── score_client.py
│
└── README.md

⚙️ How to Use the Scoring Engine
import pickle
import pandas as pd

model = pickle.load(open("models/scoring_engine.pkl", "rb"))

sample = pd.DataFrame({...})   # borrower behavioural features

score = model.predict(sample)
print("Assigned Limit:", score)

📈 Impact

70% of loan approvals automated

22% improvement in credit-risk accuracy

Fairer and more transparent limit assignment

Scalable scoring pipeline for future retraining

🧑‍💻 Author

Blaise Fonguh
Business Intelligence Intern @ CREDIX CAM S.A
