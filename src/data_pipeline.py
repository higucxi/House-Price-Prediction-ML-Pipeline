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
    # Drop columns that are mostly null or irrelevant identifiers
    drop_cols = ["Id", "Alley", "PoolQC", "Fence", "MiscFeature"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Separate numeric and categorical
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Fill numeric with median, categorical with most frequent
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

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save preprocessor for later inference
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    print(f"✅ Preprocessor created with {len(num_cols)} numeric and {len(cat_cols)} categorical features")

    # Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save processed data (as numpy arrays for simplicity)
    os.makedirs("data/processed", exist_ok=True)
    joblib.dump((X_train_processed, X_test_processed, y_train, y_test), "data/processed/train_test_split.pkl")

    print(f"✅ Data processed and saved to 'data/processed/train_test_split.pkl'")


if __name__ == "__main__":
    df = load_raw_data()
    preprocess_data(df)
