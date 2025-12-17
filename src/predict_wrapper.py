import pandas as pd
import joblib
import numpy as np

class ChurnPredictor:
    def __init__(self, model_path="models/rf_tuned_model.pkl", scaler_path="models/scaler.pkl", features_path="models/feature_names.pkl"):
        """
        Initialize the predictor by loading artifacts.
        """
        print("Loading model artifacts...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        print("Artifacts loaded.")

    def preprocess_input(self, input_data):
        """
        Replicates the preprocessing pipeline (Block 3) for a single input.
        input_data: dict
        """
        # Convert dict to DataFrame
        df = pd.DataFrame([input_data])
        
        # 1. Handle Missing Values (Same logic as training)
        if 'TotalCharges' not in df or pd.isna(df['TotalCharges'].iloc[0]):
             # We simplify here by assuming clean input or using 0, 
             # but ideally we'd load the median from training.
             df['TotalCharges'] = df['TotalCharges'].fillna(0)
             
        # 2. Feature Engineering (One-Hot Encoding)
        # We need to manually match the columns expected by the model.
        # Create a DataFrame with 0s for all model features
        df_processed = pd.DataFrame(0, index=[0], columns=self.feature_names)
        
        # Fill in numericals
        for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            if col in df.columns:
                df_processed[col] = df[col]
        
        # Fill in categoricals
        # Example: input 'InternetService'='Fiber optic' -> feature 'InternetService_Fiber optic'=1
        for col, value in input_data.items():
            if isinstance(value, str):
                feature_name = f"{col}_{value}"
                if feature_name in self.feature_names:
                    df_processed[feature_name] = 1
                    
        # 3. Scaling
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df_processed[numeric_cols] = self.scaler.transform(df_processed[numeric_cols])
        
        return df_processed

    def predict(self, input_data):
        """
        Returns probability of Churn.
        """
        X = self.preprocess_input(input_data)
        
        # Get probability of class 1 (Churn)
        prob = self.model.predict_proba(X)[0][1]
        prediction = self.model.predict(X)[0]
        
        return {
            "probability": prob,
            "churn_prediction": "Yes" if prediction == 1 else "No"
        }

if __name__ == "__main__":
    # Test Run
    predictor = ChurnPredictor()
    sample_customer = {
        'tenure': 2,
        'MonthlyCharges': 70.0,
        'TotalCharges': 150.0,
        'InternetService': 'Fiber optic',
        'Contract': 'Month-to-month',
        'PaymentMethod': 'Electronic check'
    }
    result = predictor.predict(sample_customer)
    print(f"Sample Prediction: {result}")
