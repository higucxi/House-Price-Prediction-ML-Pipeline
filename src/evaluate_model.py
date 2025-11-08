# src/evaluate_model.py

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def evaluate_model():
    # --- Load model and test data ---
    model_path = "models/model.pkl"
    data_path = "data/processed/train_test_split.pkl"

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        raise FileNotFoundError("Model or processed data not found. Run train_model.py first.")

    model = joblib.load(model_path)
    X_train, X_test, y_train, y_test = joblib.load(data_path)

    print("✅ Model and data loaded successfully.")

    # --- Generate predictions ---
    preds = model.predict(X_test)

    # --- Compute metrics ---
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n📊 Evaluation Results:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.3f}")

    # --- Visualization ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.6)
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title("Actual vs Predicted Sale Prices")
    plt.grid(True)

    # Save plot
    os.makedirs("reports", exist_ok=True)
    plot_path = "reports/actual_vs_predicted.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()

    print(f"📈 Plot saved to {plot_path}")

    # Optional: residual plot
    residuals = y_test - preds
    plt.figure(figsize=(8, 6))
    plt.scatter(preds, residuals, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted Sale Price")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.grid(True)

    residual_plot_path = "reports/residuals_plot.png"
    plt.savefig(residual_plot_path, dpi=120)
    plt.close()

    print(f"📊 Residual plot saved to {residual_plot_path}")


if __name__ == "__main__":
    evaluate_model()
