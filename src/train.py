import os
import sys
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_processing import build_pipeline, compute_rfm, assign_high_risk

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("credit_risk_experiment")

DEBUG_MODE = True
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred)
    }

def train_models(X, y, pipeline):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if DEBUG_MODE:
        models = {
            "logreg": (LogisticRegression(max_iter=200), {"C": [1]}),
            "rf": (RandomForestClassifier(max_depth=5), {"n_estimators": [10]})
        }
        cv_folds = 2
    else:
        models = {
            "logreg": (LogisticRegression(max_iter=500), {"C": [0.1, 1, 10]}),
            "rf": (RandomForestClassifier(max_depth=10), {"n_estimators": [50, 100]})
        }
        cv_folds = 3

    best_models = {}
    results = {}

    with mlflow.start_run() as run:
        for name, (model, params) in models.items():
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pkl")
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint for {name}...")
                best_model = joblib.load(checkpoint_path)
            else:
                print(f"Training {name}...")
                grid = GridSearchCV(model, params, cv=cv_folds, n_jobs=1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                joblib.dump(best_model, checkpoint_path)

            metrics = evaluate_model(best_model, X_test, y_test)
            results[name] = metrics
            best_models[name] = best_model

            print(f"\n{name} Metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

        # Select best model
        final_model_name = max(results, key=lambda m: results[m]["f1"])
        final_model = best_models[final_model_name]
        print(f"\nBest model: {final_model_name}")

        # Log parameters & metrics
        mlflow.log_param("chosen_model", final_model_name)
        for k, v in results[final_model_name].items():
            mlflow.log_metric(k, v)

        # Log preprocessor
        preprocessor_path = os.path.join(CHECKPOINT_DIR, "preprocessor.pkl")
        joblib.dump(pipeline, preprocessor_path)
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

        # Log and register the model
        mlflow.sklearn.log_model(sk_model=final_model, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "best_model")

    return final_model

if __name__ == "__main__":
    raw_data = pd.read_csv("data/raw/data (1).csv")
    raw_data.columns = raw_data.columns.astype(str)

    pipeline = build_pipeline()
    processed_features = pipeline.fit_transform(raw_data)

    snapshot_date = pd.to_datetime("2023-12-31")
    rfm = compute_rfm(raw_data, snapshot_date)
    rfm = assign_high_risk(rfm)

    processed_df = pd.DataFrame(processed_features.toarray() if hasattr(processed_features, "toarray") else processed_features)
    processed_df["CustomerId"] = raw_data["CustomerId"]
    processed_df = processed_df.merge(rfm, left_on="CustomerId", right_index=True)
    processed_df.columns = processed_df.columns.astype(str)
    X = processed_df.drop(columns=["is_high_risk", "CustomerId", "cluster"], errors="ignore")
    y = processed_df["is_high_risk"]
    print(x.head())
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    os.makedirs("data/processed", exist_ok=True)
    processed_df.to_csv("data/processed/processed_data.csv", index=False)

    train_models(X, y, pipeline)
