import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(filepath="telco_churn_data.csv"):
    """
    Performs Exploratory Data Analysis on the Telco Churn dataset.
    """
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    df = pd.read_csv(filepath)
    print("Dataset Loaded.")
    print(df.head())
    print("\nData Info:")
    print(df.info())

    # 1. Class Balance Check
    print("\n--- Class Balance (Target: Churn) ---")
    class_counts = df['Churn'].value_counts()
    print(class_counts)
    print(f"Churn Rate: {class_counts['Yes'] / len(df):.2%}")

    # Create output directory for plots
    os.makedirs("plots", exist_ok=True)

    # 2. Visualization 1: Distribution of Target Variable
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df)
    plt.title('Distribution of Churn')
    plt.savefig('plots/churn_distribution.png')
    print("Saved plots/churn_distribution.png")
    plt.close()

    # 3. Visualization 2: Correlation Heatmap (Numerical Features)
    # Convert Churn to numeric for correlation
    df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Select numerical columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_Numeric']
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('plots/correlation_heatmap.png')
    print("Saved plots/correlation_heatmap.png")
    plt.close()

    # Extra: Distribution of Tenure by Churn
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='tenure', hue='Churn', multiple="stack")
    plt.title('Tenure Distribution by Churn')
    plt.savefig('plots/tenure_by_churn.png')
    print("Saved plots/tenure_by_churn.png")
    plt.close()

if __name__ == "__main__":
    perform_eda()
