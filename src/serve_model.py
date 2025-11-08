from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(title="🏠 House Price Prediction API")

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
    MSSubClass: int
    MSZoning: str
    LotFrontage: float | None = None
    LotArea: int
    Street: str
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: str | None = None
    MasVnrArea: float | None = None
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: str | None = None
    BsmtCond: str | None = None
    BsmtExposure: str | None = None
    BsmtFinType1: str | None = None
    BsmtFinSF1: float | None = None
    BsmtFinType2: str | None = None
    BsmtFinSF2: float | None = None
    BsmtUnfSF: float | None = None
    TotalBsmtSF: float | None = None
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: str
    FirstFlrSF: int = Field(alias="1stFlrSF")
    SecondFlrSF: int = Field(alias="2ndFlrSF")
    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: int
    BsmtHalfBath: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: str | None = None
    GarageType: str | None = None
    GarageYrBlt: float | None = None
    GarageFinish: str | None = None
    GarageCars: int
    GarageArea: int
    GarageQual: str | None = None
    GarageCond: str | None = None
    PavedDrive: str
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    ThreeSsnPorch: int = Field(alias="3SsnPorch")
    ScreenPorch: int
    PoolArea: int
    PoolQC: str | None = None
    Fence: str | None = None
    MiscFeature: str | None = None
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: str
    SaleCondition: str

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
