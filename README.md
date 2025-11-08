# House-Price-Prediction-ML-Pipeline

# 🏠 House Price Prediction — End-to-End Machine Learning Pipeline

This project demonstrates a **complete machine learning workflow** built from the ground up — including data preprocessing, model training, evaluation, and deployment via a REST API.

It uses the classic **Kaggle House Prices dataset** and shows how to take a raw CSV, clean and transform it, train a regression model, evaluate its performance, and expose real-time predictions through a FastAPI endpoint.

---

## 🚀 Features

- 🧹 Data preprocessing and feature engineering  
- 🤖 Model training and saving with `scikit-learn`  
- 📊 Model evaluation with metrics and plots  
- 🌐 API endpoint for real-time predictions (`FastAPI`)  
- 🪣 MLflow experiment tracking  
- 📁 Clean project structure with modular scripts  

---


---

## ⚙️ Setup Instructions

🧹 Step 1: Run the Data Pipeline
python src/data_pipeline.py

🤖 Step 2: Train the Model
python src/train_model.py

📊 Step 3: Evaluate the Model
python src/evaluate_model.py

🌐 Step 4: Run the API Server
uvicorn src.serve_model:app --reload
The API will start on: http://127.0.0.1:8000

Either send a post request via curl, or utilize the Swagger
UI at http://127.0.0.1:8000/docs