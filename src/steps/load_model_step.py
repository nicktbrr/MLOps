import mlflow

def load_model_step(self):
    """Load the best model from MLflow"""
    print("Loading model...")
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('boston_test')
    self.model = mlflow.sklearn.load_model(f"models:/best_model/latest")