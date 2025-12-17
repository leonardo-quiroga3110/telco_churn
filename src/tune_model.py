import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, accuracy_score
import os

def tune_random_forest(data_dir="data/processed", model_dir="models"):
    """
    Antigravity Action: Hyperparameter Tuning to fix Overfitting.
    """
    print("Loading data...")
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()

    # Define Parameter Grid
    # We want to restrict the trees so they don't just memorize data.
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 8, 10],            # Limit depth to prevent overfitting
        'min_samples_leaf': [2, 4, 6],      # Require more samples per leaf (smooths boundaries)
        'class_weight': ['balanced', 'balanced_subsample'] # Handle imbalance
    }

    print("\n--- Starting GridSearchCV (Optimizing for Recall) ---")
    rf = RandomForestClassifier(random_state=42)
    
    # 5-Fold Cross-Validation emphasizing Recall
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='recall',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nBest Parameters found:", grid_search.best_params_)
    print(f"Best CV Recall Score: {grid_search.best_score_:.4f}")
    
    # Evaluate the Tuned Model
    best_rf = grid_search.best_estimator_
    
    # Check Train vs Test to see if Overfitting is checking
    train_preds = best_rf.predict(X_train)
    test_preds = best_rf.predict(X_test)
    
    train_rec = recall_score(y_train, train_preds)
    test_rec = recall_score(y_test, test_preds)
    
    print("\n--- Overfitting Check ---")
    print(f"Train Recall: {train_rec:.4f}")
    print(f"Test Recall:  {test_rec:.4f}")
    print(f"Gap:          {train_rec - test_rec:.4f}")

    # Determine if we improved
    if test_rec > 0.52:
        print("✅ Improvement achieved!")
    else:
        print("⚠️ Still struggling with Recall. Dataset signal might be weak.")

    # Save the tuned model
    joblib.dump(best_rf, f"{model_dir}/rf_tuned_model.pkl")
    print(f"Tuned model saved to {model_dir}/rf_tuned_model.pkl")

if __name__ == "__main__":
    tune_random_forest()
