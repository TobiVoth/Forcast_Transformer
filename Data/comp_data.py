from datetime import datetime
from pathlib import Path
import pandas as pd


def load_data(date_from, date_to=None):
    # Parameter sicher in Pandas Datetime umwandeln
    date_from = pd.to_datetime(date_from)
    date_to = pd.to_datetime(date_to) if date_to else pd.to_datetime(datetime.now())

    current_dir = Path(__file__).parent
    csv_path = current_dir / 'pegel_daten.csv'

    df = pd.read_csv(csv_path, sep=';')

    # format='ISO8601' liest Formate mit/ohne Sekunden flexibel ein
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')

    # Filtern nach Datum
    df = df[(df['timestamp'] >= date_from) & (df['timestamp'] <= date_to)].sort_values('timestamp')

    # Stündliche Aggregation + Interpolation
    df_hourly = (
        df.set_index('timestamp')
        .resample('1h')['value']
        .mean()
        .interpolate(method='linear')
        .reset_index()
    )

    # Export
    output_csv_path = current_dir / 'pegel_daten_comp.csv'
    df_hourly.to_csv(output_csv_path, sep=';', index=False)

    # WICHTIG: DataFrame zurückgeben!
    return df_hourly


# Aufruf
pegel_data = load_data('2000-01-01')
print(pegel_data.head())