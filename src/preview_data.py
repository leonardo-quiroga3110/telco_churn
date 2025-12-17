import pandas as pd

# Load the dataset
df = pd.read_csv('telco_churn_data.csv')

# Display settings to ensure columns aren't hidden
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("--- First 5 rows of the dataset ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Random Sample of 3 rows ---")
print(df.sample(3))
