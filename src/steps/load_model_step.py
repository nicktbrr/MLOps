import mlflow

def load_model_step(self):
    """Load the best model from MLflow"""
    print("Loading model...")
    mlflow.set_tracking_uri('https://mlops-141371485093.us-west2.run.app')
    mlflow.set_experiment('boston_test')
    self.model = mlflow.sklearn.load_model(f"models:/best_model/latest")