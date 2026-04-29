from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from datetime import datetime
import os
import sys
import joblib

sys.path.append("/opt/airflow")
from src.data_load import load_data, load_forecast_weather, load_weather_data
from src.features import cechy_pogodowe, licznik_dni_sezonu, dodaj_lagi_i_statystyki
from src.train import train_model



def feature_engineering_task():
    df = pd.read_parquet('/opt/airflow/data/tires_raw.parquet')
    df_pogoda = pd.read_parquet('/opt/airflow/data/weather_raw.parquet')

    df_final = df.merge(df_pogoda, left_index=True, right_index=True, how="left")
    df_final = cechy_pogodowe(df_final)
    df_final = licznik_dni_sezonu(df_final)
    df_final = dodaj_lagi_i_statystyki(df_final)
    
    df_final.dropna(inplace=True)

    df_final.to_parquet('/opt/airflow/data/df_final.parquet')
    print("Cechy przygotowane, zapisane do df_final.parquet")

def load_tires_wrapper(path):
    df = load_data(path)
    outputh_path = "/opt/airflow/data/tires_raw.parquet"
    df.to_parquet(outputh_path)
    print(f"Dane zapisane w {outputh_path}")

def load_weather_wrapper():
    df = load_weather_data()
    output_path = "/opt/airflow/data/weather_raw.parquet"
    df.to_parquet(output_path)
    print(f"Historia pogody zapisana w {output_path}")

def load_forecast_wrapper():
    df = load_forecast_weather()
    output_path = "/opt/airflow/data/forecast_raw.parquet"
    df.to_parquet(output_path)
    print(f"Zapisano dane prognozy pogody w {output_path}")

def split_data_wrapper():
    df = pd.read_parquet("/opt/airflow/data/df_final.parquet")
    X = df.drop(columns=["target"])
    y = df["target"]

    train_size = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

    X_train.to_parquet("/opt/airflow/data/X_train.parquet")
    X_test.to_parquet("/opt/airflow/data/X_test.parquet")
    y_train.to_frame.to_parquet("/opt/airflow/data/y_train.parquet")
    y_test.to_frame.to_parquet("/opt/airflow/data/y_test.parquet")

    print("Dane zostały podzielone i zapisane")

def train_model_wrapper():
    X_train = pd.read_parquet("/opt/airflow/data/X_train.parquet")
    X_test = pd.read_parquet("/opt/airflow/data/X_test.parquet")
    y_train = pd.read_parquet("/opt/airflow/data/y_train.parquet")["target"]
    y_test = pd.read_parquet("/opt/airflow/data/y_test.parquet")["target"]

    print("Start treningu")
    model = train_model(X_train, y_train, X_test, y_test)
    model_path = "/opt/airflow/data/lgb_model.joblib"
    joblib.dump(model, model_path)
    print("Model został zapisany w {model_path}")

with DAG(
    dag_id = "opony_prognoza_2",
    start_date = datetime(2024,1,1),
    schedule_interval=None,
    catchup=False
) as dag:
    load_csv_task = PythonOperator(
        task_id="wczytaj_dane_opon",
        python_callable=load_tires_wrapper,
        op_kwargs={'path': '/opt/airflow/data/opony_prognoza.csv'}
    )
    weather_task = PythonOperator(
        task_id="pobierz_historie_pogody",
        python_callable=load_weather_wrapper
    )
    forecast_weather_task = PythonOperator(
        task_id="pobierz_prognoze_pogody",
        python_callable=load_forecast_wrapper
    )
    feature_task = PythonOperator(
        task_id="przetwarzanie_cech",
        python_callable=feature_engineering_task
    )
    split_task = PythonOperator(
        task_id="dzielenie_df",
        python_callable=split_data_wrapper
    )
    train_task = PythonOperator(
        task_id="trenowanie_modelu",
        python_callable=train_model_wrapper
    )
    
    [load_csv_task, weather_task, forecast_weather_task] >> feature_task >> split_task >> train_task