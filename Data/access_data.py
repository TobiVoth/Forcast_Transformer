import pandas as pd
#Gibt Dataframe zurück mit Pegelständen nach Tagen.
def load_data(datum):
    # 1. Daten laden und aufbereiten
    df = pd.read_csv("Data/pegel_daten.csv", sep=';')
    df['time'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M')
    df = df[df['time'] >= datum].sort_values('time')

    # Tageswerte aggregieren und fehlende Werte interpolieren
    df_daily = df.set_index('time').resample('D')['value'].mean().interpolate(method='linear').reset_index()
    df_daily['item_id'] = 'pegel_station_1'

    return df_daily




