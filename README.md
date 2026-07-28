# 🏦 Bank Term Deposit Prediction

Predict whether a bank customer will subscribe to a **Term Deposit** using Machine Learning.

---

## 📌 Project Overview

This project aims to predict whether a client will subscribe to a **bank term deposit** based on demographic information, financial status, and previous marketing campaign interactions.

The project includes:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Modeling
- Model Evaluation
- Streamlit Web Application for real-time prediction

---

## 🎯 Problem Statement

Banks spend significant resources on marketing campaigns. Predicting customers who are more likely to subscribe to a term deposit helps improve campaign efficiency and reduce unnecessary marketing costs.

The goal is to build a classification model that predicts:

- **Yes** → Customer will subscribe
- **No** → Customer will not subscribe

---

## 📂 Project Structure

```text
📦 Bank-Term-Deposit-Prediction
│
├── final_project_LM.ipynb      # Complete data analysis & model building
├── app.py                      # Streamlit web application
├── model_pipeline.pkl          # Trained ML pipeline
├── pipeline_utils.py           # Custom preprocessing class
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The dataset contains customer information such as:

- Age
- Job
- Marital Status
- Education
- Balance
- Housing Loan
- Personal Loan
- Contact Type
- Campaign Information
- Previous Campaign Outcome
- Call Duration

**Target Variable**

- Deposit Subscription (`Yes / No`)

---

## 🛠 Data Preprocessing

The preprocessing pipeline includes:

- Removing duplicate records
- Handling missing values
- Outlier treatment using IQR Capping
- Feature Encoding
- Feature Scaling
- Building a complete preprocessing pipeline

---

## 📈 Exploratory Data Analysis

Several visualizations were performed, including:

- Target Distribution
- Age Distribution
- Balance Distribution
- Job Analysis
- Marital Status Analysis
- Education Analysis
- Campaign Analysis
- Correlation Heatmap
- Feature Relationships

---

## 🤖 Machine Learning

Several machine learning models were tested and compared.

The final selected model is:

✅ **LightGBM Classifier**

The model was chosen because it achieved the best overall performance.

---

## 📊 Model Evaluation

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score
- Confusion Matrix

---

## 🌐 Streamlit Web Application

The project includes an interactive Streamlit application where users can:

- Enter customer information
- Predict whether the customer will subscribe
- View prediction probability

Run locally:

```bash
streamlit run app.py
```

---

## 🚀 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- LightGBM
- Joblib
- Streamlit

---

## 📷 Application Preview

> Add screenshots of your Streamlit application here.

Example:

```
images/home.png
images/result.png
```

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/YourUsername/Bank-Term-Deposit-Prediction.git
```

Move to the project folder:

```bash
cd Bank-Term-Deposit-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

- Hyperparameter Optimization
- Model Explainability using SHAP
- Deploy the application online
- Add more visual analytics
- Improve UI/UX

---

## 👩‍💻 Author

**Nour**

Machine Learning & Data Science Student

---
