import sys
import os
from datetime import datetime
from airflow.sdk import dag, task   


@dag(
    dag_id="lstm_pipeline",
    start_date=datetime(2026, 2, 20),
    schedule=None,
    catchup=False,
    tags=["lstm", "sentiment"]
)
def lstm():

    @task(task_id="clean_data")
    def clean_task():
        sys.path.append("/opt/airflow/src")
        from clean_data import clean_data
        return clean_data()

    @task(task_id="preprocess_data")
    def preprocess_task():
        sys.path.append("/opt/airflow/src")
        from preprocess_data import preprocess_data
        return preprocess_data()

    @task(task_id="train_model")
    def train_task():
        sys.path.append("/opt/airflow/src")
        from train_model import train_model
        return train_model()

    @task(task_id="predict_review")
    def predict_task():
        sys.path.append("/opt/airflow/src")
        from predict_review import predict_review

        sample_review = "The movie had amazing visuals but the plot was dull."
        return predict_review(sample_review)

    clean = clean_task()
    preprocess = preprocess_task()
    train = train_task()
    predict = predict_task()

    clean >> preprocess >> train >> predict


dag = lstm()