from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title="Boston Housing Price Predictor",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

# Defining path operation for root endpoint
@app.get('/')
def main():
	return {'message': 'This is a model for classifying Reddit comments'}

@app.post('/predict')
def predict(data : dict):
	return {'result': "You win"}