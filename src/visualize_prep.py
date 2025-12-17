import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_preprocessing():
    # Load Raw and Processed Data
    raw_df = pd.read_csv("telco_churn_data.csv")
    processed_df = pd.read_csv("data/processed/X_train.csv") # Looking at training data

    print("Raw Data Shape:", raw_df.shape)
    print("Processed Data Shape:", processed_df.shape)

    # 1. Visualizing Imputation (TotalCharges)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(raw_df['TotalCharges'], kde=True, color='blue', label='Raw (With NAs)')
    plt.title('TotalCharges (Raw)')
    
    plt.subplot(1, 2, 2)
    # Reverse scaling to see the "Distribution" shape, or just plot the scaled version
    sns.histplot(processed_df['TotalCharges'], kde=True, color='green', label='Processed (Scaled)')
    plt.title('TotalCharges (Processed & Scaled)')
    
    plt.tight_layout()
    plt.savefig('plots/preprocessing_totalcharges.png')
    print("Saved plots/preprocessing_totalcharges.png")
    plt.close()

    # 2. Visualizing One-Hot Encoding
    # Compare "InternetService" (Raw) vs "InternetService_Fiber optic" (Processed)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.countplot(x='InternetService', data=raw_df)
    plt.title('InternetService Categories (Raw)')

    plt.subplot(1, 2, 2)
    # Check if column exists (it might be dropped if it was the base case, but Fiber optic usually exists)
    if 'InternetService_Fiber optic' in processed_df.columns:
        sns.countplot(x=processed_df['InternetService_Fiber optic'])
        plt.title('InternetService_Fiber optic (Encoded)')
        plt.xticks([0, 1], ['No', 'Yes'])
    else:
        print("Column InternetService_Fiber optic not found in processed data.")

    plt.tight_layout()
    plt.savefig('plots/preprocessing_encoding.png')
    print("Saved plots/preprocessing_encoding.png")
    plt.close()

    print("\n--- Processed Data Preview (First 5 Rows) ---")
    print(processed_df.head())

if __name__ == "__main__":
    visualize_preprocessing()
