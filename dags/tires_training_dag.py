import pandas as pd

def split_data_task():
    df_final = pd.read_parquet("/opt/airflow/data/df_final.parquet")
    X = df_final.drop(columns=["target"])
    y = df_final["target"]

    train_size = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

    X_train.to_parquet("opt/airflow/data/X_train.parquet")
    X_test.to_parquet("opt/airflow/data/X_test.parquet")
    y_train.to_parquet("opt/airflow/data/y_train.parquet")
    y_test.to_parquet("opt/airflow/data/y_test.parquet")

    print("Dane zostały podzielone i zapisane")