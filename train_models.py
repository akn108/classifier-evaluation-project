'''
This script will train all the 6 classification model on the same training dataset
'''
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Define the Data Source
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# Creating directory to store models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to clean the dataset
def clean_input_data(url):
    print("Data will be loaded from the url {url}")
    df = pd.read_csv(url)
    df = pd.to_numeric(df, errors='coerce')
    df.fillna(0, inplace=True)

    return df


# Function to preprocess the dataset
def preprocess_input_data(df):
    # Separating the features from the target
    if 'Churn' not in df.columns:
        raise ValueError("Dataset must contain a 'Churn' columns.")
    
    DATA_FEATURES = df.drop('Churn', axis=1)
    DATA_TARGET = df['Churn']

    # Performing Target Encoding to convert categorical values to numerical values
    lencod = LabelEncoder()
    DATA_TARGET = lencod.fit_transform(DATA_TARGET)

    # Performing Identification of Numeric Cols and Categorical Cols from the Features
    categorical_cols = DATA_FEATURES.select_dtypes(include=['object', 'category']).columns
    numerical_cols = DATA_FEATURES.select_dtypes(include=['float64','int64']).columns

    # (missed earlier) Adding One-Hot Encoding to - Converts categorical columns into binary representations
    DATA_FEATURES = pd.get_dummies(DATA_FEATURES, columns=categorical_cols, drop_first=True)

    # Performing Scaling of Numerical Features for distance based algorithms
    scaler = StandardScaler()
    DATA_FEATURES[numerical_cols] = scaler.fit_transform(DATA_FEATURES(numerical_cols))

    return DATA_FEATURES, DATA_TARGET, lencod, scaler, numerical_cols
    


def train_models():
    '''
    This will be the core function, which would perform multiple steps required for model training:
    it will perform:
    1. Getting the data loaded and clean it
    2. preprocessing the input data
    3. Splitting the data into train and test datasets
    4. Defining all the required models
    5. Train and Evaluate all defined models
    '''

    # Step 1: Getting the data loaded and clean it
    print("\n Step 1: Getting the data loaded and clean it")
    df = clean_input_data(DATA_URL)

    # Step 2: Preprocess the input data
    print("\n Step 2: Preprocess the input data")
    DATA_FEATURES, DATA_TARGET, lencod, scaler, numerical_cols = preprocess_input_data(df)
    
    ## We will save the artifacts(features/scalers/encoding etc) so that the it could be inferred
    feature_name = DATA_FEATURES.columns.to_list()
    joblib.dump(feature_name, os.path.join(MODEL_DIR, "feature_names.pkl"))
    joblib.dump(lencod, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(numerical_cols, os.path.join(MODEL_DIR, "numerical_cols.pkl"))

    # Step 3: Splitting the data into train and test datasets
    print("\n Step 3: Splitting the data into train and test datasets")
    DATA_FEATURES_Train, DATA_FEATURES_Test, DATA_TARGET_Train, DATA_TARGET_Test = train_test_split(DATA_FEATURES, DATA_TARGET, test_size=0.2, random_state=42, stratify=y)

    # Step 4: Defining all the required models
    print("\n Step 4: Defining all the required models")
    models = {
        "Logistic Regressin": LogisticRegression(max_iter1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbour": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42,use_label_encoder=False, eval_metric='logloss')
        }
    
    # Step 5: Train and evaluate all defined models
    print("\n Step 5: Train and evaluate all defined models")
    result = []
    for name, model in models.items():
        print("Training {name}...")
        model.fit(DATA_FEATURES_Train, DATA_TARGET_Train)

        print(" Saving {name}...")
        mod_name = name.replace("","_").lower()
        joblib.dump(model, os.path.join(MODEL_DIR,f"{mod_name}.pkl"))

        print("Evaluating {name}....")
        DATA_TARGET_Pred = model.predict(DATA_FEATURES_Test)
        if hasattr(model, "predict_proba"):
            DATA_TARGET_Predict_Proba = model.predict_proba(DATA_FEATURES_Test)[:,1]
        else:
            DATA_TARGET_Predict_Proba = None
        
        accur = accuracy_score(DATA_TARGET_Test, DATA_TARGET_Pred)
        prec = precision_score(DATA_TARGET_Test, DATA_TARGET_Pred)
        recall = recall_score(DATA_TARGET_Test, DATA_TARGET_Pred)
        f1 = f1_score(DATA_TARGET_Test, DATA_TARGET_Pred)
        mcc = matthews_corrcoef(DATA_TARGET_Test, DATA_TARGET_Pred)
        auc = roc_auc_score(DATA_TARGET_Test,DATA_TARGET_Predict_Proba) if DATA_TARGET_Predict_Proba is not None else 0

        result.append({
            "Model": name,
            "Accuracy": accur,
            "Precision": prec,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": mcc
        })

        print("\n--------Model COmparison Summary--------")
        summary_df = pd.DataFrame(result).set_index("Model")

        print("\nModel Training Complete. Model saved to 'models/' directory")


if __name__ == "__main__":
    train_models()

        





