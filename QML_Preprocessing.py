import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Load cleaned data
train_df = pd.read_csv("Training_BOP_cleaned.csv")
test_df = pd.read_csv("Testing_BOP_cleaned.csv")

# Step 1: Check for null values
print("Null values in Training dataset:")
print(train_df.isnull().sum())
print("\nNull values in Testing dataset:")
print(test_df.isnull().sum())

# Step 2: Select the 14 features for preprocessing
features = [
    'national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month',
    'sales_9_month', 'min_bank', 'potential_issue', 'pieces_past_due',
    'local_bo_qty', 'deck_risk', 'oe_constraint', 'ppap_risk',
    'stop_auto_buy', 'rev_stop'
]

# Ensure all features exist in the dataset
available_features = [f for f in features if f in train_df.columns]
if len(available_features) < len(features):
    missing = set(features) - set(available_features)
    print(f"Warning: The following features are missing: {missing}")

# Separate features, target, and sku
X_train = train_df[available_features]
X_test = test_df[available_features]
y_train = train_df['went_on_backorder']
y_test = test_df['went_on_backorder']
sku_train = train_df['sku']
sku_test = test_df['sku']

# Step 3: Log Transformation
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

# Remove infinity values
X_train_log = X_train_log.replace([np.inf, -np.inf], np.nan).dropna()
X_test_log = X_test_log.replace([np.inf, -np.inf], np.nan).dropna()

# Align indices
train_indices = X_train_log.index
test_indices = X_test_log.index
y_train = y_train.loc[train_indices]
y_test = y_test.loc[test_indices]
sku_train = sku_train.loc[train_indices]
sku_test = sku_test.loc[test_indices]

# Step 4: StandardScaler
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_log), columns=X_train_log.columns, index=train_indices)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_log), columns=X_test_log.columns, index=test_indices)

# Step 5: VIF Calculation
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Calculate VIF for training data
vif_data = calculate_vif(X_train_scaled)
print("\nVIF values for features:")
print(vif_data)

# Remove features with VIF > 5
while vif_data['VIF'].max() > 5:
    max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
    print(f"Removing feature {max_vif_feature} with VIF = {vif_data['VIF'].max()}")
    X_train_scaled = X_train_scaled.drop(columns=max_vif_feature)
    X_test_scaled = X_test_scaled.drop(columns=max_vif_feature)
    vif_data = calculate_vif(X_train_scaled)

# Final selected features
final_features = X_train_scaled.columns
print("\nFinal selected features after VIF filtering:")
print(final_features)

# Step 6: Combine preprocessed features, target, and sku
train_preprocessed = pd.concat([X_train_scaled, y_train.loc[train_indices]], axis=1)
train_preprocessed['sku'] = sku_train
train_preprocessed.set_index('sku', inplace=True)

test_preprocessed = pd.concat([X_test_scaled, y_test.loc[test_indices]], axis=1)
test_preprocessed['sku'] = sku_test
test_preprocessed.set_index('sku', inplace=True)

# Step 7: Save the preprocessed datasets
train_preprocessed.to_csv("Training_BOP_preprocessed.csv")
test_preprocessed.to_csv("Testing_BOP_preprocessed.csv")

print("\nPreprocessed datasets saved as:")
print("  • Training_BOP_preprocessed.csv")
print("  • Testing_BOP_preprocessed.csv")