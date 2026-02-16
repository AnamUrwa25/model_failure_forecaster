📉 MODEL FAILURE FORECASTER

An enterprise-grade Machine Learning monitoring dashboard for detecting model degradation, data drift, and prediction uncertainty.

📌 Project Overview

This project builds a universal monitoring system that works across domains such as:

⚡ Electricity load forecasting

🏦 Banking transactions

🛒 E-commerce systems

🏥 Healthcare monitoring

🔐 Cybersecurity systems

The system detects when an ML model starts failing due to:

Prediction errors increasing

Input data distribution shifting

Model uncertainty rising

⚙️ Features

✅ Train ML model (Random Forest)

📉 Measure Prediction Error (MAE)

🔀 Detect Data Drift

❓ Estimate Model Uncertainty

🧠 Compute Unified Health Score

📊 Interactive Streamlit Dashboard

📁 Downloadable Health Reports

📈 Cross-domain evaluation (Power & Bank datasets)

📊 Experiments Conducted

Power Load Forecasting Dataset

Bank Marketing Dataset

Both datasets were used to evaluate:

Model Stability

Drift Behavior

Uncertainty Behavior

Health Score consistency

🚀 How to Run
Step 1: Install Dependencies
pip install -r requirements.txt

Step 2: Run Streamlit Dashboard
streamlit run model_failure_forecaster.py

📁 Project Structure
model_failure_forecaster/
│
├── ABSTRACT.pdf
├── model_failure_forecaster.py
├── README.md
├── requirements.txt
│
├── notebooks/
│     └── model_failure_forecaster.ipynb
│
├── datasets/
│     ├── power_load_data.csv
│     └── bank.csv
│
├── reports/
│     └── model_health_report.csv

