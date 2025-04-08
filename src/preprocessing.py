import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Load the dataset
df = pd.read_csv('data/boston.csv')
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# First split into temp and test (80% / 20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Then split temp into train and validation (75% / 25%, which gives 60% / 20% / 20% overall)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42)

print(f"Train set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

# Save the original data splits
X_train.to_parquet('data/X_train.parquet')
X_val.to_parquet('data/X_val.parquet')
X_test.to_parquet('data/X_test.parquet')

# Convert Series to DataFrames for saving
pd.DataFrame(y_train, columns=['MEDV']).to_parquet('data/y_train.parquet')
pd.DataFrame(y_val, columns=['MEDV']).to_parquet('data/y_val.parquet')
pd.DataFrame(y_test, columns=['MEDV']).to_parquet('data/y_test.parquet')

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'data/scaler.pkl')

# Perform PCA with 9 components
n_components = 9
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Save the PCA model
joblib.dump(pca, 'data/pca.pkl')

# Save the PCA-transformed data
pd.DataFrame(X_train_pca).to_parquet('data/X_train_pca.parquet')
pd.DataFrame(X_val_pca).to_parquet('data/X_val_pca.parquet')
pd.DataFrame(X_test_pca).to_parquet('data/X_test_pca.parquet')
