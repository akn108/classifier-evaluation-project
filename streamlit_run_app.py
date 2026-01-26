import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import joblib
from sklearn.metrics import (accuracy_score,precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix)

# Initial Configurations
MODEL_DIR = "models"
st.set_page_config(page_title="Telco Customer Churn Prediction", layout="wide")

# Helper Functions to Load all Saved Models and required artifacts
@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        artifacts["feature_names"] = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
        artifacts["label_encoder"] = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        artifacts["scaler"] = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        artifacts["numerical_cols"] = joblib.load(os.path.join(MODEL_DIR, "numerical_cols.pkl"))

        # Making the model listing process generic    
        model_names = [
            "logistic_regressin", "decision_tree", "k-nearest_neighbour", "naive_bayes", "random_forest", "xgboost"
        ]
        for name in model_names:
            artifacts[name] = joblib.load(os.path.join(MODEL_DIR, f"{name}.pkl"))

        return artifacts
    except FileNotFoundError:
        st.error("Could not find Model file. Please check if Model Directory is correctly choosen or run the 'train_model.py' first to create the required artifacts")
        return None

# Function to preprocess using saved artifacts 
def input_preprocessing(df, artifacts):
    df = df.copy()
    # Handling Target Value if present
    DATA_TARGET = None
    if 'Churn' in df.columns:
        # Convert String Churn Values into Numeric values
        DATA_TARGET = artifacts["label_encoder"].transform(df['Churn'])
        df = df.drop(columns=['Churn'])

    # Performing One-Hot Encoding
    df_processed = pd.get_dummies(df)

    # Ensure that the input df has same columns as the trained schema
    df_processed = df_processed.reindex(columns=artifacts["feature_names"], fill_value=0)

    # Ensuring Proper Scaling of the numerical values in the dataset
    df_processed[artifacts["numerical_cols"]] = artifacts["scaler"].transform(df_processed[artifacts["numerical_cols"]])

    return df_processed, DATA_TARGET


# Implementing The Application LOgic
st.title("Telco Customer Churn Prediction System")
artifacts = load_artifacts()

if artifacts:
    st.sidebar.header("Configuration")
    st.sidebar.subheader("Get Sample Data")
    test_data_url = "https://github.com/akn108/classifier-evaluation-project/blob/main/models/test_data.csv"
    response = requests.get(test_data_url)
    csv_data = response.content
    st.sidebar.download_button(label="Download Test CSV", data=csv_data, file_name="test_data.csv", mime="text/csv")
    upload_data_file = st.sidebar.file_uploader("Upload Test Data (CSV File Only)", type=["csv"])

    model_names = {
        "Logistic Regression": "logistic_regressin",
        "Decision Tree": "decision_tree",
        "K-Nearest Neighbour": "k-nearest_neighbour",
        "Naive Bayes": "naive_bayes",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost"
    }

    select_model = st.sidebar.selectbox("Select Model", list(model_names.keys()))

    if upload_data_file:
        df_input = pd.read_csv(upload_data_file)
        st.write(f"**Loaded Dataset:** {df_input.shape} Rows, {df_input.shape[1]} Columns")

        with st.expander("View Sample Raw Data"):
            st.dataframe(df_input.head())

        if st.button("Execute Prediction"):
            # Running Pre-processing
            FEATURE_TEST, TARGET_TEST = input_preprocessing(df_input, artifacts)

            # Running Prediction
            model_key = model_names[select_model]
            model = artifacts[model_key]

            with st.spinner("Prediction in Progress..."):
                TGT_PRED = model.predict(FEATURE_TEST)
                # Run probability only for models which supports Probability. Else we will bypass
                TGT_PROB = model.predict_proba(FEATURE_TEST)[:,1] if hasattr(model, "predict_proba") else None

                if TARGET_TEST is not None:
                    # Get the Evaluation Metrics
                    st.subheader("Evaluation Metrics")
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    col1.metric("Accuracy", f"{accuracy_score(TARGET_TEST, TGT_PRED):.2f}")
                    col2.metric("Precision", f"{precision_score(TARGET_TEST, TGT_PRED):.2f}")
                    col3.metric("Recall", f"{recall_score(TARGET_TEST, TGT_PRED):.2f}")
                    col4.metric("F1 Score", f"{f1_score(TARGET_TEST, TGT_PRED):.2f}")
                    col5.metric("MCC ", f"{matthews_corrcoef(TARGET_TEST, TGT_PRED):.2f}")
                    col6.metric("AUC ", f"{roc_auc_score(TARGET_TEST, TGT_PRED):.2f}")

                    # Now Creating COnfusion Matrix 
                    with st.container():
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(TARGET_TEST, TGT_PRED)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("True")
                        st.pyplot(fig)

                else:
                    st.success("Predictions Generated")
                    df_results = df_input.copy()
                    df_results['Predicted_Churn'] = artifacts["label_encoder"].inverse_transform(TGT_PRED)
                    if TGT_PROB is not None:
                        df_results["Churn_Probability"] = TGT_PROB
                    st.dataframe(df_results)

    else:
        st.info("Please Upload CSV FIle to Proceed. Ensure that it matches the Telco Churn Schema")

