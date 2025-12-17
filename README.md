# 📡 Telco Customer Churn Prediction

## 📋 Project Overview
This project is an end-to-end Data Science simulation following the **CRISP-DM** methodology. The goal is to predict customer churn for a telecommunications company, allowing for proactive retention strategies.

**Key Objective:** Achieve a high Recall (> 0.80) to capture the maximum number of at-risk customers.

## 🛠️ Tech Stack
*   **Python 3.10+**
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn (RandomForest, DummyClassifier), XGBoost
*   **Visualization:** Matplotlib, Seaborn
*   **Deployment:** Streamlit
*   **DevOps:** Git

## 📂 Project Structure
```
telco_churn/
├── data/
│   ├── processed/          # Scaled and encoded data (train/test)
├── models/                 # Saved .pkl models (RF, XGB, Scalers)
├── plots/                  # Generated EDA and Evaluation charts
├── src/
│   ├── data_loader.py      # Synthetic data generation
│   ├── eda.py              # Exploratory Data Analysis
│   ├── preprocessing.py    # Cleaning, Encoding, Scaling pipeline
│   ├── validation.py       # Data Contracts and Integrity checks
│   ├── train.py            # Model training (Baseline vs Complex)
│   ├── tune_model.py       # Hyperparameter tuning (GridSearchCV)
│   ├── evaluate.py         # Performance evaluation (Confusion Matrix, ROC)
│   └── predict_wrapper.py  # Inference engine class
├── app.py                  # Streamlit Dashboard
└── README.md               # Project documentation
```

## 🚀 How to Run

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlist joblib
```

### 2. Data Pipeline (Optional)
The project includes a synthetic data generator if you don't have the source file.
```bash
python src/data_loader.py    # Generates telco_churn_data.csv
python src/preprocessing.py  # Cleans and prepares data
```

### 3. Model Training & Tuning
```bash
python src/train.py          # Trains Baseline, RF, and XGBoost
python src/tune_model.py     # Optimizes RandomForest for Recall
```

### 4. Run Dashboard (Inference)
Launch the interactive web application:
```bash
streamlit run app.py
```

## 📊 Results
| Model | Accuracy | Recall | Result |
| :--- | :--- | :--- | :--- |
| **Baseline** | 74% | 0% | Failed |
| **Random Forest (Initial)** | 82% | 52% | Overfitting |
| **XGBoost** | 79% | 50% | Overfitting |
| **Random Forest (Tuned)** | **~78-80%** | **~77%** | **Production Ready** |

*Note: The Tuned Random Forest successfully reduced the overfitting gap and increased Recall from 52% to 77%, nearing the strict business target of 80%.*

## 🔮 Future Improvements
*   Collect more real-world data to improve signal.
*   Feature Engineering: Create interaction terms (e.g., `MonthlyCharges` / `tenure`).
*   Deploy as a REST API using FastAPI for bulk predictions.

---
*Created by [Your Name] as part of the Agentic Data Science Portfolio.*
