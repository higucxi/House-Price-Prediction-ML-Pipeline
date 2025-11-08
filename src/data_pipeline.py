import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

def load_raw_data(path="data/raw/train.csv"):
    df = pd.read_csv(path)
    print(f"✅ Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df):
    target_col = "SalePrice"

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset. Available columns: {df.columns.tolist()[:20]}")

    # Drop high-null or irrelevant columns first
    drop_cols = ["Id", "Alley", "PoolQC", "Fence", "MiscFeature"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Separate target early to avoid including it in feature lists
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Identify numeric and categorical columns from features only
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Pipelines
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save everything
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    os.makedirs("data/processed", exist_ok=True)
    joblib.dump((X_train_processed, X_test_processed, y_train, y_test), "data/processed/train_test_split.pkl")

    print("✅ Data preprocessing complete.")
    print(f"   X_train shape: {X_train_processed.shape}")
    print(f"   X_test shape:  {X_test_processed.shape}")



if __name__ == "__main__":
    df = load_raw_data()
    preprocess_data(df)
