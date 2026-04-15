import pandas as pd
import matplotlib.pyplot as plt


def predict_model(df: pd.DataFrame, model, X_train, df_history):
    prognozy = []
    future_dates = df.index
    for data in future_dates:
        # Pobierz cechy pogodowe dla dnia
        wiersz_dzis = df.loc[[data]].copy()
        
        # Dynamiczna aktualizacja lagów
        wiersz_dzis["lag_1"] = df_history['target'].iloc[-1]
        wiersz_dzis["lag_7"] = df_history['target'].iloc[-7]
        wiersz_dzis["lag_14"] = df_history['target'].iloc[-14]
        
        # Lag sprzed roku (bezpieczne wyszukiwanie)
        data_rok_temu = data - pd.DateOffset(years=1)
        if data_rok_temu in df.index:
            wiersz_dzis["lag_365"] = df.loc[data_rok_temu, 'target']
        else:
            wiersz_dzis["lag_365"] = df_history['target'].iloc[-365] if len(df_history) >= 365 else 0

        # Przeliczenie statystyk kroczących (okno przesuwa się z każdą prognozą)
        wiersz_dzis["window_7_std"] = df_history['target'].iloc[-7:].std()
        wiersz_dzis["window_14_std"] = df_history['target'].iloc[-14:].std()
        wiersz_dzis["window_28_std"] = df_history['target'].iloc[-28:].std()

        # Dopasowanie kolumn do modelu
        wiersz_dzis = wiersz_dzis[X_train.columns]
        
        # Predykcja
        pred = model.predict(wiersz_dzis)[0]
        if pred < 0: pred = 0
        prognozy.append(pred)
        
        # Dodanie prognozy do historii (dla następnego kroku pętli)
        nowy_wpis = wiersz_dzis.copy()
        nowy_wpis['target'] = pred
        df_history = pd.concat([df_history, nowy_wpis])
        
    finalna_prognoza = pd.Series(prognozy, index=future_dates)
    return prognozy, finalna_prognoza


def create_forecast_plot(df_final, finalna_prognoza):
    plt.figure(figsize=(12, 6))
    df_final['target'].tail(30).plot(label='Historia (ostatnie 30 dni)', color='blue')
    finalna_prognoza.plot(label='Prognoza (następne 14 dni)', color='red', linestyle='--', marker='o')
    plt.title('Prognoza sprzedaży opon')
    plt.xlabel('Data')
    plt.ylabel('Sztuki')
    plt.legend()
    plt.grid(True)
    plt.savefig("wykres_prognozy.png")
    print("Wykres został zapisany jako wykres_prognozy.png")
    plt.show()