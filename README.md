# House-Price-Prediction-ML-Pipeline

Smart column handling: drops columns like Alley and PoolQC that are mostly missing (based on the description file).

Automatic categorical encoding: uses OneHotEncoder for categorical fields instead of manually creating dummies.

Stores preprocessing logic: saves a fitted preprocessor so the same logic is used at inference time.

Joblib serialization: saves processed train/test sets for model training.

Integrate a Streamlit front-end for visualization

Add model drift detection