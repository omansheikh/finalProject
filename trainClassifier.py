import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load data
print("Loading data from crop_production.csv...")
df = pd.read_csv('crop_production.csv')
print(f"Loaded {len(df)} rows of data")

"""
Data measurements correction:
- Area: Hectares (ha)
- Production: Kilograms (kg)
- Yield = Production/Area = Kilograms per Hectare (kg/ha)
"""

# Clean the data
print("\nCleaning data...")
# Replace any missing values in categorical columns with 'Unknown'
categorical_columns = ['State_Name', 'District_Name', 'Season', 'Crop']
for col in categorical_columns:
    missing = df[col].isna().sum()
    df[col] = df[col].fillna('Unknown')
    print(f"Filled {missing} missing values in {col}")

# Convert Area to numeric, replacing invalid values with the median
print("\nProcessing Area column (in hectares)...")
df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
area_nulls = df['Area'].isna().sum()
area_median = df['Area'].median()
df['Area'] = df['Area'].fillna(area_median)
print(f"Fixed {area_nulls} invalid/missing Area values")

# Convert Production to numeric and remove any negative values
print("\nProcessing Production column (in kg)...")
df['Production'] = pd.to_numeric(df['Production'], errors='coerce')
initial_rows = len(df)
df = df[df['Production'] >= 0]
df = df.dropna(subset=['Production'])

# Calculate yield in kg/hectare
df['Yield'] = df['Production'] / df['Area']
# Remove unrealistic yields (more than 100,000 kg/ha or 100 tonnes/ha is unlikely)
df = df[df['Yield'] <= 100000]
print(f"Removed {initial_rows - len(df)} rows with invalid/unrealistic Production values")
print(f"Final dataset size: {len(df)} rows")

# Create and fit the OneHotEncoder for categorical columns
print("\nEncoding categorical features...")
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
ohe.fit(df[categorical_columns])
print("Categorical encoding completed")

# Transform categorical columns
print("Transforming features...")
encoded_features = ohe.transform(df[categorical_columns])
print(f"Generated {encoded_features.shape[1]} encoded features")

# Create feature matrix X and target y
print("\nPreparing final feature matrix...")
X = np.hstack([encoded_features, df[['Area']].values])
y = df['Yield'].values  # Now predicting yield (kg/ha) instead of total production
print(f"Final feature matrix shape: {X.shape}")

# Train Random Forest model with optimized parameters
print("\nTraining Random Forest model...")
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=20,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Starting model training (this may take a few minutes)...")
model.fit(X, y)
print("Model training completed")

# Save model and encoder
print("\nSaving model and encoder...")
with open('crop_yield_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('oneHotEncoder.pkl', 'wb') as f:
    pickle.dump(ohe, f)

print("\nModel and encoder saved successfully!")
print("\nNote: The model expects Area in hectares and predicts Yield in kg/hectare")