📉Model Failure Forecaster

Enterprise grade machine learning monitoring dashboard that detects model degradation, data drift, and prediction uncertainty before performance visibly declines.

📌Overview

This project provides a universal monitoring framework that works across multiple domains such as electricity forecasting, banking systems, ecommerce platforms, healthcare monitoring, and cybersecurity applications.

The system identifies early warning signals when a model begins to fail due to rising prediction error, shifting input distributions, or increasing uncertainty.

📌Features

Train Random Forest model
Measure prediction error using MAE
Detect data drift
Estimate model uncertainty
Compute unified health score
Interactive Streamlit dashboard
Downloadable health reports
Cross domain evaluation

📌Experiments

Datasets used

Power Load Forecasting dataset
Bank Marketing dataset

Evaluated metrics

Model stability
Drift behavior
Uncertainty behavior
Health score consistency

📌Run Locally

Install dependencies

pip install -r requirements.txt


Launch dashboard

streamlit run model_failure_forecaster.py

📁Project Structure
model_failure_forecaster/
│
├── ABSTRACT.pdf
├── model_failure_forecaster.py
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── model_failure_forecaster.ipynb
│
├── datasets/
│   ├── power_load_data.csv
│   └── bank.csv
│
└── reports/
    └── model_health_report.csv
