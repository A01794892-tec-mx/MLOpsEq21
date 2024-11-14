from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel
from typing import List
import mlflow.pyfunc
import pandas as pd

# MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "fire_forest_LR_model"  # Replace with your registered model name

# Global variable for the model
model = None

# Function to fetch the model by version number
def fetch_latest_model():
    global model
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        client = mlflow.MlflowClient()
        model_versions = client.search_registered_models(f"name='{MODEL_NAME}'")
        print(f"model_versions : {model_versions}")

        # Access the latest version (most recent version) in the model's `latest_versions` field
        latest_version = model_versions[0].latest_versions[0]  # Get the first model's latest version
        print(f'latest version: {latest_version.version}')

        model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"

        # Load the model using its URI
        model = mlflow.pyfunc.load_model(model_uri)

        print(f"Loaded model: {MODEL_NAME}, version: {latest_version.version}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# Define the lifespan for FastAPI
def lifespan(app: FastAPI):
    print("Starting up application...")
    fetch_latest_model()
    yield  # Application runs here
    print("Shutting down application...")

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Endpoint to reload the latest model manually
@app.post("/reload_model")
def reload_model():
    try:
        fetch_latest_model()
        return {"message": "Model reloaded successfully"}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define the input data format
class ForestFiresData(BaseModel):
    features: List[List[float]]  # Batch of records

# Prediction endpoint
@app.post("/predict")
def predict(fire_forest_data: ForestFiresData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    # Convert input to DataFrame
    try:
        input_data = pd.DataFrame(fire_forest_data.features)
        predictions = model.predict(input_data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "MLflow FastAPI is running!"}
