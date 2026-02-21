# src/train_model.py

import os
import pickle
import yaml
import shutil
from datetime import datetime

import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


PARAMS_PATH = "/opt/airflow/params.yml"
MODEL_ROOT = "/opt/airflow/models"


def train_model():

    # =========================
    # Load Parameters (Airflow Safe)
    # =========================
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)

    temp_preprocessed_path = os.path.join(MODEL_ROOT, "preprocessed_temp.pkl")

    if not os.path.exists(temp_preprocessed_path):
        raise FileNotFoundError("Preprocessed file not found.")

    model_name = params["training"]["registered_model_name"]
    experiment_name = params["training"]["experiment_name"]

    # =========================
    # MLflow Setup (DagsHub)
    # =========================
    username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

    mlflow.set_tracking_uri(
        f"https://{username}:{password}@dagshub.com/sableen-kaur788/mlflow-dvc-airflow.mlflow"
    )

    mlflow.set_experiment(experiment_name)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    # Auto-increment run name based on existing runs
    existing_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    run_number = len(existing_runs) + 1
    run_name = f"LSTM_v{run_number}"

    # =========================
    # Create Local Version Folder
    # =========================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    version_dir = os.path.join(MODEL_ROOT, f"{timestamp}_v{run_number}")
    os.makedirs(version_dir, exist_ok=True)

    # =========================
    # Load Preprocessed Data
    # =========================
    with open(temp_preprocessed_path, "rb") as f:
        X_train, X_test, y_train, y_test, tokenizer, le, max_len = pickle.load(f)

    vocab_size = min(
        params["preprocessing"]["max_features"],
        len(tokenizer.word_index) + 1
    )

    # =========================
    # Build Model
    # =========================
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=params["training"]["embedding_dim"]
        ),
        LSTM(
            params["training"]["lstm_units"],
            dropout=params["training"]["dropout"]
        ),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    # =========================
    # Start MLflow Run
    # =========================
    with mlflow.start_run(run_name=run_name):

        print(f"Starting MLflow run: {run_name}")

        # Log Hyperparameters
        mlflow.log_params({
            "embedding_dim": params["training"]["embedding_dim"],
            "lstm_units": params["training"]["lstm_units"],
            "dropout": params["training"]["dropout"],
            "epochs": params["training"]["epochs"],
            "batch_size": params["training"]["batch_size"]
        })

        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=params["training"]["epochs"],
            batch_size=params["training"]["batch_size"],
            verbose=1
        )

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        y_pred_prob = model.predict(X_test, verbose=0).ravel()
        y_pred_class = (y_pred_prob > 0.5).astype(int)

        # Safe metric logging
        precision = precision_score(y_test, y_pred_class, zero_division=0)
        recall = recall_score(y_test, y_pred_class, zero_division=0)
        f1 = f1_score(y_test, y_pred_class, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_pred_prob)
        except Exception:
            auc = 0.0

        mlflow.log_metric("test_accuracy", float(test_acc))
        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("precision", float(precision))
        mlflow.log_metric("recall", float(recall))
        mlflow.log_metric("f1", float(f1))
        mlflow.log_metric("auc", float(auc))

        # Register model in MLflow Model Registry
        mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name
        )

    # =========================
    # Save Local Artifacts
    # =========================
    model.save(os.path.join(version_dir, "lstm_model.h5"))

    with open(os.path.join(version_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    with open(os.path.join(version_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    shutil.move(
        temp_preprocessed_path,
        os.path.join(version_dir, "preprocessed.pkl")
    )

    print(f"[train_model] Saved model version at {version_dir}")

    return version_dir