<<<<<<< HEAD
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model
model = joblib.load("sales_forecasting_model.pkl")
model_features = joblib.load("model_features.pkl")

# Define input structure
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


@app.post("/predict")
def predict(data: SalesInput):
    df = pd.DataFrame([data.dict()])
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_features, fill_value=0)

    prediction = model.predict(df)

    return {"predicted_sales": float(prediction[0])}
=======
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model
model = joblib.load("sales_forecasting_model.pkl")
model_features = joblib.load("model_features.pkl")

# Define input structure
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


@app.post("/predict")
def predict(data: SalesInput):
    df = pd.DataFrame([data.dict()])
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_features, fill_value=0)

    prediction = model.predict(df)

    return {"predicted_sales": float(prediction[0])}
>>>>>>> 296fe8a (Initial commit)
