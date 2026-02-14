# Telco Customer Churn Prediction

## Problem Statement
In the highly competitive telecommunications industry, customer retention is critical for profitability. Acquiring a new customer is significantly more expensive than retaining an existing one. "Churn" refers to the phenomenon where customers discontinue their service. The goal of this project is to develop a machine learning system that can accurately predict which customers are at risk of churning based on their demographic, account, and service usage data. By identifying these at-risk customers early, the company can proactively engage them with retention strategies, thereby reducing revenue loss and improving customer loyalty.

## Dataset Description

**Dataset Name:** Telco Customer Churn (Telco-Customer-Churn.csv)

**Source:**  IBM/Github Source - Sample Data Sets

**Description:**
The dataset contains **7,043 instances** (rows) and **21 features** (columns). Each row represents a unique customer.

* Target Variable: **Churn (Yes/No)** - Indicates if the customer left within the last month.

* Features:

    * **Demographics:** gender, SeniorCitizen, Partner, Dependents.

    * **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies.

    * **Account Information:** tenure (months stayed), Contract (Month-to-month, One year, Two year), PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges.

## Models Used
The following six supervised machine learning models were implemented and evaluated on the test set (20% split).

###  Comparison Table with the evaluation metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Row 1, Col 2 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 |
| Decision Tree | Row 1, Col 2 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 |
| K-NN | Row 1, Col 2 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 |
| Naive Bayes | Row 1, Col 2 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 |
| Random Forest (Ensemble) | Row 1, Col 2 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 |
| XGBoost (Ensemble)  | Row 1, Col 2 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 | Row 1, Col 3 |


### Observations on the performance of each model
| ML Model Name | Observation about model performance | 
| --- | --- | 
| Logistic Regression | Row 1, Col 2 |
| Decision Tree | Row 1, Col 2 |
| K-NN | Row 1, Col 2 |
| Naive Bayes | Row 1, Col 2 |
| Random Forest (Ensemble) |
| XGBoost (Ensemble)  | Row 1, Col 2 |

## The Project(Code) Structure
```
classifier-evaluation-project/
├── streamlit_run_app.py      # Main Streamlit application
├── train_models.py           # Script to train all models
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── model/                    # Directory for saved models
│   ├── decision_tree.pkl     # Trained model files
|   ├── feature_names.pkl     # Trained model files
|   ├── k-nearest_neighbour.pkl     # Trained model files
|   ├── label_encoder.pkl     # Trained model files
|   ├── logistic_regressin.pkl     # Trained model files
|   ├── naive_bayes.pkl     # Trained model files
|   ├── random_forest.pkl     # Trained model files
|   ├── xgboost.pkl     # Trained model files
│   ├── scaler.pkl           # Feature scaler
│   ├── numerical_cols.json         # Evaluation metrics
│   ├── test_data.csv        # Test dataset

```

