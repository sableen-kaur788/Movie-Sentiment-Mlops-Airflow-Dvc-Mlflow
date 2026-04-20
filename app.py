import os
import pickle

import yaml
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to params.yml (used for max_len, etc.)
PARAMS_PATH = os.path.join(BASE_DIR, "params.yml")

# Root folder where versioned model directories are stored (DVC-managed)
MODEL_ROOT = os.path.join(BASE_DIR, "models")


def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def get_latest_model_dir():
    """
    Return the most recently modified subdirectory inside MODEL_ROOT.
    This mirrors the logic in src/predict_review.py but for local paths.
    """
    if not os.path.isdir(MODEL_ROOT):
        raise FileNotFoundError(
            f"No models directory found at {MODEL_ROOT}. "
            f"Make sure you've pulled models (e.g. via DVC) or copied them here."
        )

    subdirs = [
        os.path.join(MODEL_ROOT, d)
        for d in os.listdir(MODEL_ROOT)
        if os.path.isdir(os.path.join(MODEL_ROOT, d))
    ]

    if not subdirs:
        raise FileNotFoundError(
            f"No trained model versions found inside {MODEL_ROOT}."
        )

    return max(subdirs, key=os.path.getmtime)


def load_artifacts(model_dir: str):
    """
    Load LSTM model, tokenizer, and label encoder from a specific model directory.
    Expected files inside model_dir:
        - lstm_model.h5
        - tokenizer.pkl
        - label_encoder.pkl
    """
    model_path = os.path.join(model_dir, "lstm_model.h5")
    tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
    label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder file not found at {label_encoder_path}")

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    model = safe_load_h5_model(model_path, tokenizer)

    return model, tokenizer, label_encoder


def _build_compatible_model(tokenizer):
    """
    Rebuild the same architecture used in src/train_model.py,
    then load weights from the saved H5 file.
    """
    vocab_size = min(
        params["preprocessing"]["max_features"],
        len(tokenizer.word_index) + 1,
    )

    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(
            input_dim=vocab_size,
            output_dim=params["training"]["embedding_dim"],
        ),
        LSTM(
            params["training"]["lstm_units"],
            dropout=params["training"]["dropout"],
        ),
        Dense(1, activation="sigmoid"),
    ])
    return model


def safe_load_h5_model(model_path: str, tokenizer):
    """
    Load a .h5 Keras model with backward-compatibility fallback for
    quantization_config deserialization errors.
    """
    try:
        return load_model(model_path)
    except Exception as exc:
        error_text = str(exc)
        if "quantization_config" in error_text or "Could not locate class 'Sequential'" in error_text:
            print(
                "[app.py] Detected Keras/H5 compatibility issue; "
                "rebuilding model architecture and loading weights."
            )

            rebuilt_model = _build_compatible_model(tokenizer)
            rebuilt_model.load_weights(model_path)
            return rebuilt_model

        raise


# =========================
# App initialization
# =========================

params = load_params()
max_len = params["preprocessing"]["max_len"]

# Resolve the model directory dynamically (latest version)
model_dir = get_latest_model_dir()
model, tokenizer, label_encoder = load_artifacts(model_dir)

app = Flask(__name__)


def preprocess_text(text: str):
    """
    Convert raw text to padded sequence using the fitted tokenizer.
    """
    if text is None:
        return None

    cleaned = text.strip().lower()
    if not cleaned:
        return None

    seq = tokenizer.texts_to_sequences([cleaned])
    if not seq or not seq[0]:
        # No known tokens
        return None

    padded = pad_sequences(seq, maxlen=max_len)
    return padded


def predict_review(text: str):
    """
    Predict the sentiment for a single review string using the loaded artifacts.
    Returns a dict with review, sentiment label, and confidence score.
    """
    seq = preprocess_text(text)
    if seq is None:
        return {
            "error": "Input text contains no known tokens for this tokenizer.",
            "review": text,
        }

    pred_prob = model.predict(seq, verbose=0).ravel()[0]
    pred_binary = int(pred_prob > 0.5)
    pred_label = label_encoder.inverse_transform([pred_binary])[0]

    return {
        "review": text,
        "sentiment": pred_label,
        "confidence": float(pred_prob),
    }


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        text = request.form.get("review")
        if text:
            result = predict_review(text)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    # For local testing
    app.run(debug=True)

