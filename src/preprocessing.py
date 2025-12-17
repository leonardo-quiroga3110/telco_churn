import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from validation import check_data_contract, check_cleanliness

def preprocess_data(input_path="telco_churn_data.csv", output_dir="data/processed"):
    """
    Antigravity Action: Cleaning, Encoding, Scaling.
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    
    # 1. Validation (Contract Check)
    print("--- Executing Data Contract Checks ---")
    check_data_contract(df)

    # 2. Handling Missing Values (Imputation)
    # Strategy: TotalCharges often has NaNs when tenure=0. Fill with 0 or Median.
    print(f"NAs before cleaning: {df['TotalCharges'].isnull().sum()}")
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # MLOps Validation
    check_cleanliness(df)

    # 3. Handling Target Variable
    # Convert Yes/No to 1/0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drop ID (not a feature)
    df = df.drop(columns=['customerID'])

    # 4. Feature Engineering: Encoding Categoricals
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"Categorical columns to encode: {list(categorical_cols)}")
    
    # Using One-Hot Encoding via get_dummies for simplicity and interpretability
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 5. Scaling Numerical Features
    # Identify numerical columns (excluding Target)
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    
    print("Feature Scaling Applied (StandardScaler).")

    # Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Split Data
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save processed data
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    # Save Scaler for inference later (MLOps)
    joblib.dump(scaler, "models/scaler.pkl")
    # Save columns list to ensure correct order during inference
    joblib.dump(X_train.columns.tolist(), "models/feature_names.pkl")

    print(f"Preprocessing Complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Artifacts saved to {output_dir} and models/")

if __name__ == "__main__":
    preprocess_data()
