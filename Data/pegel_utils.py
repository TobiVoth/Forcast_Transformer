import pandas as pd
import matplotlib.pyplot as plt
from _datetime import datetime, timedelta

#Gibt Dataframe zurück mit Pegelständen nach Tagen.
def load_data(date_from, date_to=None):
    if date_to is None:
        date_to = datetime.now()

    #Daten laden und aufbereiten
    df = pd.read_csv("Data/pegel_daten.csv", sep=';')
    df['time'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M')
    df = df[df['time'] >= date_from].sort_values('time')
    df = df[df['time'] <= date_to]
    #Tageswerte aggregieren
    df_daily = df.set_index('time').resample('D')['value'].mean().interpolate(method='linear').reset_index()

    #aktivieren, falls Autogluon genutzt werden soll
    #df_daily['item_id'] = 'pegel_station_1'

    return df_daily

def plot_pegel_data(pegel_df, date_from, date_to=None):
    date_from = datetime.strptime(date_from, "%Y-%m-%d")


    if date_to is None:
        date_to = datetime.now()
    else:
        date_to = datetime.strptime(date_to, "%Y-%m-%d")

    if date_from == date_to:
        date_to = date_to + timedelta(days=1)

    pegel_df = pegel_df[pegel_df['time'] >= date_from].sort_values('time')
    pegel_df = pegel_df[pegel_df['time'] <= date_to]


    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Zeit (time)')
    ax1.set_ylabel('Pegelstand (value)', color=color)
    ax1.plot(pegel_df['time'], pegel_df['value'], color=color, label='Pegelstand')
    ax1.tick_params(axis='y', labelcolor=color)

    plt.title('Pegelstand im Zeitraum')
    fig.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()