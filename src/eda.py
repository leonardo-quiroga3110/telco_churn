import pandas as pd
import numpy as np
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

    # Set professional style
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    custom_palette = ["#B0BEC5", "#1E88E5"] # Grey for No, Blue for Yes
    
    # Create output directory for plots
    os.makedirs("plots", exist_ok=True)

    # 2. Visualization 1: Distribution of Target Variable
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='Churn', data=df, palette=custom_palette)
    plt.title('Distribution of Customer Churn', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Churn Status', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add percentage labels
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_height() + 50
        ax.annotate(percentage, (x, y), fontsize=12, fontweight='bold')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig('plots/churn_distribution.png', dpi=300)
    print("Saved plots/churn_distribution.png")
    plt.close()

    # 3. Visualization 2: Correlation Heatmap (Numerical Features)
    # Convert Churn to numeric for correlation
    df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Select numerical columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_Numeric']
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdBu', fmt=".2f", 
                linewidths=0.5, cbar_kws={"shrink": .8}, vmin=-1, vmax=1)
    plt.title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png', dpi=300)
    print("Saved plots/correlation_heatmap.png")
    plt.close()

    # Extra: Distribution of Tenure by Churn
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='tenure', hue='Churn', fill=True, palette=custom_palette, alpha=0.6, linewidth=0)
    plt.title('Tenure Distribution by Churn Status', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Tenure (Months)')
    plt.ylabel('Density')
    sns.despine()
    plt.tight_layout()
    plt.savefig('plots/tenure_by_churn.png', dpi=300)
    print("Saved plots/tenure_by_churn.png")
    plt.close()

if __name__ == "__main__":
    perform_eda()
