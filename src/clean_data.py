import pandas as pd
import os
import yaml

PARAMS_PATH = "/opt/airflow/params.yml"
with open(PARAMS_PATH) as f:
    params = yaml.safe_load(f)

RAW_PATH = os.path.join("/opt/airflow", params['data']['raw_path'])
CLEANED_PATH = os.path.join("/opt/airflow", params['data']['cleaned_path'])

def clean_data():
    df = pd.read_csv(RAW_PATH)
    df = df.drop_duplicates().dropna()
    os.makedirs(os.path.dirname(CLEANED_PATH), exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)
    print(f"[clean_data] Data cleaned at {CLEANED_PATH}")
    return CLEANED_PATH