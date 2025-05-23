from metaflow import step
import mlflow
    

def register_model_step(self):
    """Register the best model with MLflow"""
    mlflow.set_tracking_uri('https://mlops-141371485093.us-west2.run.app')
    mlflow.set_experiment('boston')
    mlflow.sklearn.log_model(self.best_model, artifact_path='regression_models', registered_model_name="best_model")