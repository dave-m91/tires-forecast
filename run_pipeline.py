from model_prognozy.src.data_load import load_data, load_weather_data, load_forecast_weather
from model_prognozy.src.features import cechy_pogodowe, licznik_dni_sezonu, dodaj_lagi_i_statystyki
from model_prognozy.src.predict import predict_model, create_forecast_plot
from model_prognozy.src.train import train_model


df = load_data("data/opony_prognoza.csv")
df_pogoda = load_weather_data()

df_final = df.merge(df_pogoda, left_index=True, right_index=True, how="left")
df_final = cechy_pogodowe(df_final)
df_final = licznik_dni_sezonu(df_final)
df_final = dodaj_lagi_i_statystyki(df_final)
df_final.dropna(inplace=True)

X = df_final.drop(columns=['target'])
y = df_final['target']

train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

model = train_model(X_train, y_train, X_test, y_test)

df_future_pogoda = load_forecast_weather()
df_history = df_final.copy()

df_future_input = cechy_pogodowe(df_future_pogoda)
df_future_input = licznik_dni_sezonu(df_future_input)

prognozy, finalna_prognoza = predict_model(df_future_input, model, X_train, df_history)

create_forecast_plot(df_final, finalna_prognoza)