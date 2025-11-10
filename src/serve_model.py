from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(title="🏠 House Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React dev server
        "http://127.0.0.1:3000"    # alternative local address
    ],
    allow_credentials=True,
    allow_methods=["*"],   # allow all HTTP methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],   # allow all headers
)

# --- Load model and preprocessor ---
model_path = "models/model.pkl"
preprocessor_path = "models/preprocessor.pkl"

if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
    raise FileNotFoundError("Trained model or preprocessor not found. Run training first.")

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

print("✅ Model and preprocessor loaded successfully.")

# --- Define input schema for requests ---
class HouseFeatures(BaseModel):
    MSSubClass: Optional[int] = None
    MSZoning: Optional[str] = None
    LotFrontage: Optional[float] = None
    LotArea: Optional[int] = None
    Street: Optional[str] = None
    LotShape: Optional[str] = None
    LandContour: Optional[str] = None
    Utilities: Optional[str] = None
    LotConfig: Optional[str] = None
    LandSlope: Optional[str] = None
    Neighborhood: Optional[str] = None
    Condition1: Optional[str] = None
    Condition2: Optional[str] = None
    BldgType: Optional[str] = None
    HouseStyle: Optional[str] = None
    OverallQual: Optional[int] = None
    OverallCond: Optional[int] = None
    YearBuilt: Optional[int] = None
    YearRemodAdd: Optional[int] = None
    RoofStyle: Optional[str] = None
    RoofMatl: Optional[str] = None
    Exterior1st: Optional[str] = None
    Exterior2nd: Optional[str] = None
    MasVnrType: Optional[str] = None
    MasVnrArea: Optional[float] = None
    ExterQual: Optional[str] = None
    ExterCond: Optional[str] = None
    Foundation: Optional[str] = None
    BsmtQual: Optional[str] = None
    BsmtCond: Optional[str] = None
    BsmtExposure: Optional[str] = None
    BsmtFinType1: Optional[str] = None
    BsmtFinSF1: Optional[float] = None
    BsmtFinType2: Optional[str] = None
    BsmtFinSF2: Optional[float] = None
    BsmtUnfSF: Optional[float] = None
    TotalBsmtSF: Optional[float] = None
    Heating: Optional[str] = None
    HeatingQC: Optional[str] = None
    CentralAir: Optional[str] = None
    Electrical: Optional[str] = None
    FirstFlrSF: Optional[int] = Field(default=None, alias="1stFlrSF")
    SecondFlrSF: Optional[int] = Field(default=None, alias="2ndFlrSF")
    LowQualFinSF: Optional[int] = None
    GrLivArea: Optional[int] = None
    BsmtFullBath: Optional[int] = None
    BsmtHalfBath: Optional[int] = None
    FullBath: Optional[int] = None
    HalfBath: Optional[int] = None
    BedroomAbvGr: Optional[int] = None
    KitchenAbvGr: Optional[int] = None
    KitchenQual: Optional[str] = None
    TotRmsAbvGrd: Optional[int] = None
    Functional: Optional[str] = None
    Fireplaces: Optional[int] = None
    FireplaceQu: Optional[str] = None
    GarageType: Optional[str] = None
    GarageYrBlt: Optional[float] = None
    GarageFinish: Optional[str] = None
    GarageCars: Optional[int] = None
    GarageArea: Optional[int] = None
    GarageQual: Optional[str] = None
    GarageCond: Optional[str] = None
    PavedDrive: Optional[str] = None
    WoodDeckSF: Optional[int] = None
    OpenPorchSF: Optional[int] = None
    EnclosedPorch: Optional[int] = None
    ThreeSsnPorch: Optional[int] = Field(default=None, alias="3SsnPorch")
    ScreenPorch: Optional[int] = None
    PoolArea: Optional[int] = None
    PoolQC: Optional[str] = None
    Fence: Optional[str] = None
    MiscFeature: Optional[str] = None
    MiscVal: Optional[int] = None
    MoSold: Optional[int] = None
    YrSold: Optional[int] = None
    SaleType: Optional[str] = None
    SaleCondition: Optional[str] = None

    class Config:
        allow_population_by_field_name = True

@app.post("/predict")
def predict(features: HouseFeatures):
    # Convert request JSON → DataFrame
    data = pd.DataFrame([features.model_dump(by_alias=True)])

    # Apply same preprocessing as training
    processed = preprocessor.transform(data)

    # Predict
    prediction = model.predict(processed)[0]
    return {"predicted_price": round(float(prediction), 2)}

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}

if __name__ == "__main__":
    uvicorn.run("src.serve_model:app", host="0.0.0.0", port=8000, reload=True)
