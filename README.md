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
| Logistic Regression | 0.80 | 0.71 | 0.63 | 0.54 | 0.59 | 0.45 |
| Decision Tree | 0.77 | 0.68 | 0.57 | 0.51 | 0.54 | 0.38 |
| K-NN |  0.77 | 0.70 | 0.56 | 0.56 | 0.56 | 0.40 |
| Naive Bayes | 0.28 | 0.50 | 0.26 | 0.95 | 0.41 | -0.02 |
| Random Forest (Ensemble) | 0.80 | 0.69 | 0.68 | 0.46 | 0.55 |0.44 |
| XGBoost (Ensemble)  |0.79 | 0.71 | 0.61 | 0.54 | 0.57 | 0.43 |


### Observations on the performance of each model
| ML Model Name | Observation about model performance | 
| --- | --- | 
| Logistic Regression | Accuracy is 0.80 and AUC is 0.71, which shows good overall performance. Precision (0.63) and recall (0.54) are fairly balanced. Easy to understand model, but it misses some churn cases. |
| Decision Tree | Accuracy is 0.77 and AUC is 0.68, a bit lower than Logistic Regression. Precision (0.57) and recall (0.51) are weaker, meaning more mistakes. Tends to overfit, so not very reliable. |
| K-NN | Accuracy is 0.77 and AUC is 0.70, similar to Decision Tree. Precision and recall are both 0.56, so it’s balanced but average. Needs careful tuning of K value to improve results. |
| Naive Bayes | Accuracy is very low (0.28) but recall is very high (0.95). This means it predicts almost everyone as churn, but precision is poor (0.26). MCC is negative (-0.02), showing weak correlation. Not suitable for this dataset. |
| Random Forest (Ensemble) |Accuracy is 0.80, same as Logistic Regression. Precision is high (0.68) but recall is low (0.46), so it correctly predicts churn when it says so, but misses many actual churners. Good if false positives are costly.|
| XGBoost (Ensemble)  | Accuracy is 0.79 and AUC is 0.71, close to Logistic Regression. Precision (0.61) and recall (0.54) are balanced. Handles complex data well and is a strong candidate for deployment. |

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

