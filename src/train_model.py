import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

def train_model():
    # --- Load preprocessed data ---
    data_path = "data/processed/train_test_split.pkl"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Processed data not found at {data_path}. "
            "Run src/data_pipeline.py first."
        )

    X_train, X_test, y_train, y_test = joblib.load(data_path)
    print(f"✅ Loaded processed data — X_train: {X_train.shape}, X_test: {X_test.shape}")

    # --- Initialize model ---
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    # --- Train model ---
    print("🚀 Training RandomForestRegressor...")
    model.fit(X_train, y_train)

    # --- Evaluate model ---
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n✅ Model training complete.")
    print(f"📊 Mean Absolute Error (MAE): {mae:.2f}")
    print(f"📉 Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"📈 R² Score: {r2:.3f}")

    # --- Log with MLflow ---
    mlflow.set_experiment("house-price-prediction")
    with mlflow.start_run():
        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": 200,
            "max_depth": None,
            "random_state": 42
        })
        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })
        mlflow.sklearn.log_model(model, "random_forest_model")
        print("🧾 Metrics and model logged to MLflow.")

    # --- Save trained model ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("💾 Model saved to models/model.pkl")


if __name__ == "__main__":
    train_model()

