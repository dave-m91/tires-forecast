import mlflow.lightgbm
import streamlit as st
import mlflow
import matplotlib.pyplot as plt
import numpy as np

from model_prognozy.src.data_load import load_data, load_weather_data, load_forecast_weather
from model_prognozy.src.features import cechy_pogodowe, licznik_dni_sezonu, dodaj_lagi_i_statystyki
from model_prognozy.src.predict import predict_model
from model_prognozy.src.train import train_model

from model_prognozy.utilites import get_last_run_date, features_explainer


st.set_page_config(page_title="Predykcja sprzedaży opon", layout="wide")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
#run_id = "0f65e1c0dce34f7eb0b0b3cdef25ecbc"
experiment = mlflow.get_experiment_by_name("Tire Forecast")
exp_id = experiment.experiment_id if experiment else "0"
df_runs = mlflow.search_runs(
    experiment_ids=[exp_id],
    filter_string="status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=1
)
if df_runs.empty:
    st.error("Nie znaleziono modelu w MLflow. Uruchom trenowanie")
    st.stop()
run_id = df_runs.iloc[0].run_id
model_uri = f"runs:/{run_id}/lightgbm"
loaded_model = mlflow.lightgbm.load_model(model_uri)

run_data = mlflow.get_run(run_id).data

rmse = run_data.metrics.get("rmse")
r2 = run_data.metrics.get("R2")

try:
    df = load_data("data/opony_prognoza.csv")
    df_pogoda = load_weather_data()

    df_final = df.merge(df_pogoda, left_index=True, right_index=True, how="left")
    df_final = cechy_pogodowe(df_final)
    df_final = licznik_dni_sezonu(df_final)
    df_final = dodaj_lagi_i_statystyki(df_final)
    df_final.dropna(inplace=True)
    df_history = df_final.copy()
    df_final_last30 = df_final.copy()
    df_final_last30 = df_final_last30.tail(30)
    df_final_last30 = df_final_last30[["target"]]
except:
    print("Problem przy ładowaniu danych")

X = df_final.drop(columns=['target'])
y = df_final['target']
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

df_future_pogoda = load_forecast_weather()
df_future_input = cechy_pogodowe(df_future_pogoda)
df_future_input = licznik_dni_sezonu(df_future_input)

prognozy, finalna_prognoza = predict_model(df_future_input, model = loaded_model,
                                        X_train=X_train, df_history=df_history)

st.title("Prognoza sprzedaży opon w kolejnych 14 dniach")

with st.sidebar:
    st.header("Wytrenuj model")
    if st.button("Uruchom trenowanie"):
        with st.spinner("Trwa trenowanie modelu..."):
            try:
                model = train_model(X_train, y_train, X_test, y_test)
                st.success("Trenowanie zakończone")
                st.rerun()
            except Exception as e:
                st.error(f"Błąd podczas trenowania: {e}")
    last_date = get_last_run_date(run_id)
    st.caption(f"Ostatnie szkolenie: {last_date}")
            

tab1, tab2 = st.tabs(["Prognoza modelu", "Istotność cech"])

with tab1:

    col_1, col_2 = st.columns(2)
    with col_1:
        st.metric("RMSE", value=np.round(rmse,2))
    with col_2:
        st.metric("R2", value=np.round(r2,2))

    fig, ax = plt.subplots(figsize=(12,6), dpi=300)
    df_final['target'].tail(30).plot(ax=ax,label='Historia (ostatnie 30 dni)', color='blue')
    finalna_prognoza.plot(ax=ax, label='Prognoza (następne 14 dni)', color='red', linestyle='--', marker='o')
    ax.set_title('Prognoza sprzedaży opon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, width="stretch")

with tab2:
    fig = features_explainer(loaded_model, X_train)
    st.pyplot(fig, width="stretch")

