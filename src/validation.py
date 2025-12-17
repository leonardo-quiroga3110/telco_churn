import pandas as pd

def check_data_contract(df):
    """
    PM Action: Define Data Contract.
    Enforces that specific columns exist and checks for data integrity.
    """
    required_columns = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]
    
    # Check 1: Existence of required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Data Contract Violated: Missing columns {missing_cols}")
    
    # Check 2: Data Types (Basic check)
    assert pd.api.types.is_numeric_dtype(df['tenure']), "Tenure must be numeric"
    assert pd.api.types.is_numeric_dtype(df['MonthlyCharges']), "MonthlyCharges must be numeric"
    
    print("✅ Data Contract Check 1 Passed: Structure Validated.")

def check_cleanliness(df):
    """
    MLOps Practice: Validation after cleaning.
    """
    # Assert no missing values remain
    null_counts = df.isnull().sum().sum()
    assert null_counts == 0, f"Cleanliness Violated: Dataset still has {null_counts} missing values!"
    print(f"✅ Data Contract Check 2 Passed: Cleanliness Validated (0 NAs).")

if __name__ == "__main__":
    # Test on raw data (Check 1 should pass, Check 2 might fail if run on raw)
    df = pd.read_csv("telco_churn_data.csv")
    check_data_contract(df)
