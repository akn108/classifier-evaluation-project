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

    # Performing Scaling of Numerical Features for distance based algorithms
    scaler = StandardScaler()
    DATA_FEATURES[numerical_cols] = scaler.fit_transform(DATA_FEATURES(numerical_cols))

    return DATA_FEATURES, DATA_TARGET, lencod, scaler, numerical_cols
    