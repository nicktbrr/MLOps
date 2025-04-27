import mlflow
import numpy as np

def evaluate_step(self):
    """Evaluate the model"""
    print("Evaluating model...")
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('boston_test')
    with mlflow.start_run():
        y_val_pred = self.model.predict(self.X_val_pca)
        val_mse = np.mean((self.y_val - y_val_pred) ** 2)
        val_rmse = np.sqrt(val_mse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mse", val_mse)