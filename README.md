# Phishing URL Detection System

## Overview
This project uses **machine learning** to detect phishing websites based on URL features. Given a URL, the model predicts whether it is **Safe** or **Phishing**.

---

## Features
- Detects phishing URLs using a trained ML model.
- Simple and easy to use in Python.
- Uses features like URL length, presence of `@`, IP address usage, and more.

---

## Technology Stack
- **Language:** Python
- **Machine Learning Library:** Scikit-learn
- **Model Storage:** Pickle (`.pkl`)

---
Dataset
Based on Kaggle Phishing Websites Dataset
Features include:
URL length
Presence of @ symbol
HTTPS usage
Presence of IP address
Domain features, etc.

Future Enhancements
Add more features for better accuracy.
Experiment with other ML models like Random Forest or XGBoost.
Automate feature extraction from raw URLs.
