import pandas as pd
import requests
import datetime as dt
import streamlit as st

@st.cache_data(ttl=43200)
def load_data(path):
    print("Ładowanie danych")
    df = pd.read_csv(path, decimal=",")
    df["Day [Date] PE-D01"] = pd.to_datetime(df["Day [Date] PE-D01"])

    # Uzupełnianie brakujących dat zerami
    pelny_zakres = pd.date_range(start=df["Day [Date] PE-D01"].min(), end=df["Day [Date] PE-D01"].max())
    df.set_index("Day [Date] PE-D01", inplace=True)
    df = df.reindex(pelny_zakres).fillna(0)
    df.index.name = 'date'
    df.rename(columns={'Units Sold ST-010': 'target'}, inplace=True)
    return df

@st.cache_data(ttl=43200)
def load_weather_data():
    print("Pobieranie danych pogodowych")
    przedwczoraj = dt.datetime.now() - dt.timedelta(days=2)
    wczoraj = dt.datetime.now() - dt.timedelta(days=1)
    przedwczoraj = przedwczoraj.strftime("%Y-%m-%d")
    wczoraj = wczoraj.strftime("%Y-%m-%d")
    url_pogoda = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 52.2298,
        "longitude": 21.0118,
        "start_date": "2021-01-01",
        "end_date": przedwczoraj,
        "daily": ["temperature_2m_mean", "cloud_cover_mean", "wind_gusts_10m_mean", "wind_speed_10m_mean", "relative_humidity_2m_mean", "precipitation_sum", "snowfall_sum"],
        "timezone": "Europe/Berlin"
    }
    response = requests.get(url_pogoda, params=params)

    if response.status_code == 200:
        data = response.json()
        dates = data["daily"]["time"]
        temp_avg = data["daily"]["temperature_2m_mean"]
        opady = data["daily"]["precipitation_sum"]
        zachmurzenie = data["daily"]["cloud_cover_mean"]
        wiatr = data["daily"]["wind_speed_10m_mean"]
        podmuchy_wiatru = data["daily"]["wind_gusts_10m_mean"]
        wilgotnosc = data["daily"]["relative_humidity_2m_mean"]
        opady_sniegu = data["daily"]["snowfall_sum"]


        df_pogoda = pd.DataFrame({
            "date": dates,
            "temp_mean": temp_avg,
            "opady": opady,
            "zachmurzenie": zachmurzenie,
            "wiatr": wiatr,
            "podmuchy_wiatru": podmuchy_wiatru,
            "wilgotnosc": wilgotnosc,
            "opady_sniegu": opady_sniegu
        })

        df_pogoda = df_pogoda.explode("date")
        df_pogoda["date"] = pd.to_datetime(df_pogoda["date"])
        df_pogoda = df_pogoda.sort_values(by="date")
        df_pogoda.set_index("date", inplace=True)
        df_pogoda.fillna(0, inplace=True)
        
        return df_pogoda
        
    else:
        print(f"Błąd podczas pobierania danych. Kod odpowiedzi: {response.status_code}")


@st.cache_data(ttl=43200)
def load_forecast_weather():
    print("Pobieranie danych pogodowych z przyszłosci")
    url_forecast = "https://api.open-meteo.com/v1/forecast"
    params_f = {
        "latitude": 52.2298,
        "longitude": 21.0118,
        "daily": ["temperature_2m_mean", "cloud_cover_mean", 
                  "wind_gusts_10m_mean", "wind_speed_10m_mean", 
                  "relative_humidity_2m_mean", "precipitation_sum", "snowfall_sum"],
        "timezone": "Europe/Berlin",
        "forecast_days": 14
    }

    resp_f = requests.get(url_forecast, params=params_f)
    if resp_f.status_code == 200:
        resp_f = resp_f.json()
        df_future_pogoda = pd.DataFrame({
            "date": pd.to_datetime(resp_f["daily"]["time"]),
            "temp_mean": resp_f["daily"]["temperature_2m_mean"],
            "opady": resp_f["daily"]["precipitation_sum"],
            "zachmurzenie": resp_f["daily"]["cloud_cover_mean"],
            "wiatr": resp_f["daily"]["wind_speed_10m_mean"],
            "podmuchy_wiatru": resp_f["daily"]["wind_gusts_10m_mean"],
            "wilgotnosc": resp_f["daily"]["relative_humidity_2m_mean"],
            "opady_sniegu": resp_f["daily"]["snowfall_sum"]
            }).set_index("date")
        return df_future_pogoda
    else:
        print(f"Błąd podczas pobierania danych. Kod odpowiedzi: {resp_f.status_code}")
        