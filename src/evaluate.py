import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_models(data_dir="data/processed", model_dir="models"):
    """
    Antigravity Action: Evaluate Models on Test Set.
    """
    # Load Test Data (The 200 customers the model barely knows)
    print("Loading Test data...")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()
    
    # Load Models
    baseline = joblib.load(f"{model_dir}/baseline_model.pkl")
    rf_model = joblib.load(f"{model_dir}/rf_model.pkl")
    xgb_model = joblib.load(f"{model_dir}/xgb_model.pkl")
    
    models = {
        "Baseline": baseline,
        "RandomForest": rf_model,
        "XGBoost": xgb_model
    }
    
    results = []
    
    print("\n--- Evaluation Results (Test Set) ---")
    
    for name, model in models.items():
        preds = model.predict(X_test)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds, zero_division=0)
        
        # Calculate AUC (Baseline might not support predict_proba well or returns 0s)
        try:
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
        except:
            auc = 0.5 # Baseline default
            
        print(f"\nModel: {name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Recall:   {rec:.4f} (Target >= 0.80)")
        print(f"ROC-AUC:  {auc:.4f}")
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Recall": rec,
            "AUC": auc
        })

        # Plot Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues', cbar=False,
                    annot_kws={'size': 14, 'weight': 'bold'})
        plt.title(f'Confusion Matrix\n{name}', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('Actual Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'plots/confusion_matrix_{name.lower()}.png', dpi=300)
        plt.close()

    # Feature Importance (RandomForest)
    print("\n--- Extracting Feature Importance (RF) ---")
    feature_names = joblib.load(f"{model_dir}/feature_names.pkl")
    importances = rf_model.feature_importances_
    
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.set_context("notebook", font_scale=1.1)
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette="viridis")
    plt.title('Top 10 Feature Drivers of Churn', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300)
    print("Saved plots/feature_importance.png")
    
    # Save Results CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv("evaluation_results.csv", index=False)
    print("\nSaved evaluation_results.csv")

if __name__ == "__main__":
    evaluate_models()
