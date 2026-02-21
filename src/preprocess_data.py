# src/preprocess_data.py

import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import yaml

PARAMS_PATH = "/opt/airflow/params.yml"

with open(PARAMS_PATH) as f:
    params = yaml.safe_load(f)

CLEANED_PATH = os.path.join("/opt/airflow", params['data']['cleaned_path'])
MODEL_DIR = "/opt/airflow/models"
os.makedirs(MODEL_DIR, exist_ok=True)


def preprocess_data():
    print("[preprocess_data] Loading cleaned data...")
    df = pd.read_csv(CLEANED_PATH)

    # ==========================
    # 🔹 CLEAN LABELS PROPERLY
    # ==========================
    print("[preprocess_data] Cleaning sentiment labels...")
    df['sentiment'] = (
        df['sentiment']
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print("Unique labels after cleaning:", df['sentiment'].unique())

    # ==========================
    # 🔹 ENSURE BINARY ONLY
    # ==========================
    unique_labels = df['sentiment'].unique()

    if len(unique_labels) != 2:
        raise ValueError(
            f"Expected 2 classes but found {len(unique_labels)}: {unique_labels}"
        )

    # ==========================
    # 🔹 TEXT PREPARATION
    # ==========================
    texts = df['review'].astype(str).tolist()

    tokenizer = Tokenizer(
        num_words=params['preprocessing']['max_features'],
        oov_token="<OOV>"
    )
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_len = params['preprocessing']['max_len']
    X = pad_sequences(sequences, maxlen=max_len)

    # ==========================
    # 🔹 LABEL ENCODING
    # ==========================
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])

    print("Encoded classes:", le.classes_)

    # ==========================
    # 🔹 TRAIN TEST SPLIT
    # ==========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params['data']['test_size'],
        random_state=params['data']['random_state'],
        stratify=y   # VERY IMPORTANT
    )

    # ==========================
    # 🔹 SAVE TEMP FILE
    # ==========================
    temp_path = os.path.join(MODEL_DIR, "preprocessed_temp.pkl")

    with open(temp_path, "wb") as f:
        pickle.dump(
            (X_train, X_test, y_train, y_test, tokenizer, le, max_len),
            f
        )

    print(f"[preprocess_data] Saved temp preprocessed file at {temp_path}")
    return temp_path