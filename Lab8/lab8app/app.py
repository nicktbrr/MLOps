from fastapi import FastAPI
import mlflow
import uvicorn
import pandas as pd
app = FastAPI(
    title="Boston Housing Price Predictor",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

mlflow_path = 'mlflow.db'

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'This is a model for classifying Reddit comments'}

@app.post('/predict')
def predict(data : dict):
	mlflow.set_tracking_uri(f"sqlite:///{mlflow_path}")
	X = data['data']
	X = pd.DataFrame(X)
	print(X)
	model = mlflow.sklearn.load_model("models:/boston/1")
	preds = model.predict(X)
	print(preds)
	return {'result': preds.tolist()}