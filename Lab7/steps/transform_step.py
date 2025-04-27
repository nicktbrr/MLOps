from metaflow import step
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def transform_step(self):
    """Transform and split data"""
    print("Transforming and splitting data...")
    X = self.data.drop('MEDV', axis=1)
    y = self.data['MEDV']

    # First split into temp and test (80% / 20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Then split temp into train and validation (75% / 25%, which gives 60% / 20% / 20% overall)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    self.X_train = X_train_scaled
    self.X_val = X_val_scaled
    self.y_train = y_train
    self.y_val = y_val
    self.X_test = X_test_scaled
    self.y_test = y_test
