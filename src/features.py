import pandas as pd
import numpy as np
import holidays

def cechy_pogodowe(df_input):
    df_feat = df_input.copy()
    # Cechy cykliczne
    """df_feat["day_sin"] = np.sin(2*np.pi*df_feat.index.dayofweek / 7)
    df_feat["day_cos"] = np.cos(2*np.pi*df_feat.index.dayofweek / 7)
    week = df_feat.index.isocalendar().week.astype(float)
    df_feat["week_sin"] = np.sin(2*np.pi*week / 52)
    df_feat["week_cos"] = np.cos(2*np.pi*week / 52)
    df_feat["month_sin"] = np.sin(2*np.pi*df_feat.index.month / 12)
    df_feat["month_cos"] = np.cos(2*np.pi*df_feat.index.month / 12)"""
    
    # Święta
    pl_holidays = holidays.Poland()
    # Zabezpieczenie przed upewnieniem się, że indeks jest typu datetime przed odczytaniem właściwości date
    dates = pd.to_datetime(df_feat.index).date
    df_feat["is_holiday"] = [1 if d in pl_holidays else 0 for d in dates]
    
    # Logika biznesowa (np. chłód)
    df_feat["chlod"] = np.where(df_feat["temp_mean"] < 7, 1, 0)
    return df_feat


def licznik_dni_sezonu(df):
    df["sezon_zimowy"] = 0
    obecny_sezon_zimowy = 0
    for i, row in df.iterrows():
        miesiac = i.month
        if miesiac >= 9 and miesiac <= 11:
            obecny_sezon_zimowy += 1
            df.loc[i, "sezon_zimowy"] = obecny_sezon_zimowy
        else:
            obecny_sezon_zimowy = 0
    return df

def dodaj_lagi_i_statystyki(df, col='target'):
    df = df.copy()
    df["lag_1"] = df[col].shift(1)
    df["lag_7"] = df[col].shift(7)
    df["lag_14"] = df[col].shift(14)
    df["lag_365"] = df[col].shift(365)
    df["window_7_std"] = df[col].shift(1).rolling(window=7).std()
    df["window_14_std"] = df[col].shift(1).rolling(window=14).std()
    df["window_28_std"] = df[col].shift(1).rolling(window=28).std()
    return df