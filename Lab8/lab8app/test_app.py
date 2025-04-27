import requests
import pandas as pd
import mlflow

url = 'http://localhost:8000/predict'

df = pd.read_csv('../../data/boston.csv')
df1 = df.drop(columns='MEDV')

data = df1.iloc[0:2,:].to_dict(orient='records')

X = {
    'data': data
}

print(X)

try:
    response = requests.post(url, json=X)
    print(response.json())
except Exception as e:
    print(e)