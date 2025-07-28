import pandas as pd

# Load data
train_df = pd.read_csv("dataset/Training_BOP.csv")
test_df = pd.read_csv("dataset/Testing_BOP.csv")

def preprocess(df):
    # Step 1: Check for negative values in perf_6_month_avg and perf_12_month_avg, and remove these columns if found
    columns_to_drop = []
    if (df['perf_6_month_avg'] < 0).any():
        columns_to_drop.append('perf_6_month_avg')
    if (df['perf_12_month_avg'] < 0).any():
        columns_to_drop.append('perf_12_month_avg')
    
    if columns_to_drop:
        print(f"Removing columns with negative values: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    
    # Step 2: Remove rows with '?' in any column
    df = df.replace('?', pd.NA).dropna()
    
    # Step 3: Convert 'Yes'/'No' categorical columns to binary (1/0)
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        if set(df[col].dropna().unique()).issubset({'Yes', 'No'}):
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    return df

# Apply preprocessing
train_clean = preprocess(train_df)
test_clean = preprocess(test_df)

# Save cleaned datasets
train_clean.to_csv("Training_BOP_cleaned.csv", index=False)
test_clean.to_csv("Testing_BOP_cleaned.csv", index=False)

print("✅ Preprocessing complete. Cleaned files saved as:")
print("  • Training_BOP_cleaned.csv")
print("  • Testing_BOP_cleaned.csv")