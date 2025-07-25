import joblib
import mlflow.pyfunc
import pandas as pd

# Load preprocessor
preprocessor = joblib.load("checkpoints/preprocessor.pkl")

# Load model from MLflow
model_uri = "models:/best_model/Production"
model = mlflow.pyfunc.load_model(model_uri)

# Prepare new raw data
new_data = pd.read_csv("data/raw/new_customers.csv")
new_data.columns = new_data.columns.astype(str)

# Preprocess the new data
processed = preprocessor.transform(new_data)

# Convert to DataFrame if needed
processed_df = pd.DataFrame(processed.toarray() if hasattr(processed, "toarray") else processed)

# Predict high-risk customers
predictions = model.predict(processed_df)

# Attach predictions to original data
new_data["is_high_risk_prediction"] = predictions
new_data.to_csv("data/predicted/high_risk_predictions.csv", index=False)

print(new_data[["CustomerId", "is_high_risk_prediction"]].head())
