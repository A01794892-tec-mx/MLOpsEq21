from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import uvicorn
#from sklearn.datasets import load_wine

# Load the saved model
with open("modelLR_v1.0.6.pkl", "rb") as f:
    model = pickle.load(f)

# Load target names for response
#data = load_wine()
#target_names = data.target_names

# Define the input data format for prediction
#class WineData(BaseModel):
#    features: List[float]
class ForestFiresData(BaseModel):
    features: List[float]

# Initialize FastAPI app
app = FastAPI()

# Define prediction endpoint
@app.post("/predict")
def predict(fire_forest_data: ForestFiresData):
    # Validate input length
    #if len(wine_data.features) != model.n_features_in_:
    #    raise HTTPException(
    #        status_code=400,
    #        detail=f"Input must contain {model.n_features_in_} features."
    #    )

    # Make prediction
    prediction = model.predict([fire_forest_data.features])[0]
    #prediction_name = target_names[prediction]
    
    return {"prediction": prediction}

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Team 21 Model API"}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)