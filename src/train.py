import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, accuracy_score
import os

def train_models(data_dir="data/processed", model_dir="models"):
    """
    Antigravity Action: Train Baseline and Complex Models.
    """
    # Load Processed Data
    print("Loading training data...")
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
    
    # 1. Baseline Model (The "Floor")
    # Strategy: predicting the majority class (No Churn)
    print("\n--- Training Baseline Model (Dummy) ---")
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train, y_train)
    
    # Quick check on Train (Real evaluation is in Block 5)
    base_preds = baseline.predict(X_train)
    print(f"Baseline Train Accuracy: {accuracy_score(y_train, base_preds):.4f}")
    # Recall will likely be 0.0 because it predicts "No" for everyone and we care about "Yes"
    print(f"Baseline Train Recall: {recall_score(y_train, base_preds, zero_division=0):.4f}")
    
    # 2. Complex Model (The "Solution")
    # Using RandomForest with class_weight='balanced' to optimize for Recall as requested by PM
    print("\n--- Training Complex Model (RandomForest) ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    rf_preds = rf_model.predict(X_train)
    print(f"RandomForest Train Accuracy: {accuracy_score(y_train, rf_preds):.4f}")
    print(f"RandomForest Train Recall: {recall_score(y_train, rf_preds):.4f}")

    # 3. XGBoost Model (The "Gradient Boosting" Approach)
    print("\n--- Training XGBoost Model ---")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    xgb_preds = xgb_model.predict(X_train)
    print(f"XGBoost Train Accuracy: {accuracy_score(y_train, xgb_preds):.4f}")
    print(f"XGBoost Train Recall: {recall_score(y_train, xgb_preds):.4f}")
    
    # 4. Model Versioning (MLOps)
    print("\n--- Saving Models ---")
    os.makedirs(model_dir, exist_ok=True)
    
    baseline_path = f"{model_dir}/baseline_model.pkl"
    rf_path = f"{model_dir}/rf_model.pkl"
    xgb_path = f"{model_dir}/xgb_model.pkl"
    
    joblib.dump(baseline, baseline_path)
    joblib.dump(rf_model, rf_path)
    joblib.dump(xgb_model, xgb_path)
    
    print(f"Baseline saved to: {baseline_path}")
    print(f"RandomForest saved to: {rf_path}")
    print(f"XGBoost saved to: {xgb_path}")

if __name__ == "__main__":
    train_models()
