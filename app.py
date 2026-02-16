import os
import requests
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ==============================
# Download Model from Drive
# ==============================

MODEL_URL = "https://drive.google.com/uc?export=download&id=1tH_kemSLO9-wzjRl5ZUNfks2hepWSILb"
MODEL_PATH = "sales_forecasting_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

# ==============================
# Load Model + Features
# ==============================

model = joblib.load(MODEL_PATH)
model_features = joblib.load("model_features.pkl")

# ==============================
# Input Schema
# ==============================

class SalesInput(BaseModel):
    Item_Weight: float
    Item_Fat_Content: str
    Item_Visibility: float
    Item_Type: str
    Item_MRP: float
    Outlet_Size: str
    Outlet_Location_Type: str
    Outlet_Type: str
    Outlet_Age: int


# ==============================
# Prediction Endpoint
# ==============================

@app.post("/predict")
def predict(data: SalesInput):
    df = pd.DataFrame([data.dict()])
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_features, fill_value=0)

    prediction = model.predict(df)

    return {"predicted_sales": float(prediction[0])}


# ==============================
# Root Check Endpoint
# ==============================

@app.get("/")
def home():
    return {"message": "Sales Forecast API is running 🚀"}
