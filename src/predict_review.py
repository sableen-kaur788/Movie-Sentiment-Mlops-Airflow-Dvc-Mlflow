# src/predict_review.py
import os
import pickle
import yaml
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

PARAMS_PATH = "/opt/airflow/params.yml"

with open(PARAMS_PATH) as f:
    params = yaml.safe_load(f)

MODEL_ROOT = "/opt/airflow/models"


def get_latest_model_dir():
    subdirs = [
        os.path.join(MODEL_ROOT, d)
        for d in os.listdir(MODEL_ROOT)
        if os.path.isdir(os.path.join(MODEL_ROOT, d))
    ]

    if not subdirs:
        raise Exception("No trained models found.")

    return max(subdirs, key=os.path.getmtime)


def predict_review(text: str):

    model_dir = get_latest_model_dir()

    model = load_model(os.path.join(model_dir, "lstm_model.h5"))

    with open(os.path.join(model_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        seq,
        maxlen=params['preprocessing']['max_len']
    )

    pred_prob = model.predict(padded, verbose=0).ravel()[0]
    pred_binary = int(pred_prob > 0.5)
    pred_label = le.inverse_transform([pred_binary])[0]

    return {
        "review": text,
        "sentiment": pred_label,
        "confidence": float(pred_prob)
    }