import pandas as pd
import numpy as np
import random

def generate_telco_data(n_samples=1000):
    """
    Generates a synthetic Telco Customer Churn dataset.
    """
    np.random.seed(42)
    random.seed(42)

    data = {
        'customerID': [f'{random.randint(1000,9999)}-{random.choice(["A","B","C"])}{random.randint(100,999)}' for _ in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.20]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'MonthlyCharges': np.random.uniform(18.25, 118.75, n_samples),
    }

    df = pd.DataFrame(data)
    
    # TotalCharges is roughly tenure * MonthlyCharges (with some noise and handling missing for 0 tenure)
    df['TotalCharges'] = df['tenure'] * df['MonthlyCharges'] + np.random.normal(0, 10, n_samples)
    df['TotalCharges'] = df['TotalCharges'].apply(lambda x: max(0, x)) # No negative charges
    
    # Introduce some NAs to simulate real world messiness (as per Block 3 requirements)
    df.loc[np.random.choice(df.index, 20), 'TotalCharges'] = np.nan

    # Generate Churn based on some rules to make it learnable
    # Higher churn for Month-to-month, Fiber optic, Electronic check
    # Lower churn for high tenure, Two year contract
    
    churn_prob = np.zeros(n_samples)
    churn_prob += np.where(df['Contract'] == 'Month-to-month', 0.4, 0)
    churn_prob += np.where(df['Contract'] == 'Two year', -0.3, 0)
    churn_prob += np.where(df['InternetService'] == 'Fiber optic', 0.2, 0)
    churn_prob += np.where(df['tenure'] < 12, 0.2, 0)
    churn_prob += np.where(df['tenure'] > 60, -0.2, 0)
    churn_prob += np.random.normal(0, 0.2, n_samples) # Noise
    
    # Sigmoid-ish to probability
    churn_prob = 1 / (1 + np.exp(-(churn_prob - 0.5))) # Shift center
    
    df['Churn'] = ['Yes' if p > 0.5 else 'No' for p in churn_prob]
    
    return df

if __name__ == "__main__":
    df = generate_telco_data()
    print(f"Generated dataset with shape: {df.shape}")
    df.to_csv("telco_churn_data.csv", index=False)
    print("Saved to telco_churn_data.csv")
