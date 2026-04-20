# Sentiment Analysis MLOps Pipeline with Airflow, DVC, MLflow, and LSTM

An end-to-end MLOps project for movie review sentiment analysis using an LSTM model, orchestrated with Apache Airflow, tracked with MLflow (DagsHub), and versioned with DVC.

## What This Project Does

- Cleans raw review data.
- Tokenizes and preprocesses text into padded sequences.
- Trains an LSTM sentiment model.
- Logs experiments and metrics to MLflow.
- Versions model artifacts with DVC.
- Serves predictions through a Flask web app.

## Tools and Technologies Used

### Core Language and Libraries

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn
- scikit-learn
- PyYAML
- Joblib

### Deep Learning / NLP

- TensorFlow
- Keras
- `Tokenizer` + `pad_sequences` for text preprocessing
- LSTM network for binary sentiment classification

### MLOps and Pipeline

- Apache Airflow (DAG orchestration)
- MLflow (experiment tracking and model registry)
- DagsHub MLflow tracking backend
- DVC (data/model artifact versioning)

### App and Infrastructure

- Flask (inference web app)
- Docker
- Docker Compose
- PostgreSQL (Airflow metadata DB)
- Redis (Celery broker)
- Celery Executor (Airflow workers)

## Pipeline Flow

The Airflow DAG `lstm_pipeline` runs tasks in this order:

1. `clean_data`
2. `preprocess_data`
3. `train_model`
4. `predict_review` (sample prediction task)


<img width="1907" height="960" alt="1" src="https://github.com/user-attachments/assets/9e376086-b983-4af4-ba3b-f1afe469b2e6" />


<img width="1906" height="875" alt="3" src="https://github.com/user-attachments/assets/04048c9c-7b83-44bc-84fe-3ab49e736fff" />

Each training run creates a versioned folder under `models/` with:

- `lstm_model.h5`
- `tokenizer.pkl`
- `label_encoder.pkl`
- `preprocessed.pkl`

## Configuration

Main config file: `params.yml`

Key settings include:

- Data paths (`raw_path`, `cleaned_path`)
- Train/test split (`test_size`, `random_state`)
- Text preprocessing (`max_features`, `max_len`)
- LSTM hyperparameters (`embedding_dim`, `lstm_units`, `dropout`, `epochs`, `batch_size`)
- MLflow metadata (`experiment_name`, `registered_model_name`)

## Local Setup

### 1) Create environment and install dependencies

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Prepare models/data artifacts

If using DVC-managed artifacts:

```bash
dvc pull
```

This should populate the `models/` directory tracked by `models.dvc`.

### 3) Run Flask app

```bash
python app.py
```

Then open:

- http://127.0.0.1:5000/

## Running with Airflow (Docker)

Build and start services:

```bash
docker compose up --build
```

Access:

- Airflow UI: http://localhost:8080

From Airflow UI, trigger DAG:

- `lstm_pipeline`

## MLflow Tracking

Training logs and metrics are sent to the configured DagsHub MLflow tracking URI in `src/train_model.py`.

Typical logged metrics:

- `test_accuracy`
- `test_loss`
- `precision`
- `recall`
- `f1`
- `auc`

## Notes

- `app.py` loads the latest model version from `models/`.
- If `models/` is missing, pull artifacts with DVC or copy model folders manually.
- The Flask template displays `sentiment` and `confidence` from inference output.

## Future Improvements

- Add unit/integration tests for preprocessing and inference.
- Add API endpoint (`/predict`) for programmatic access.
- Add CI workflow for lint/test/build checks.
- Add better error telemetry and model health checks.



