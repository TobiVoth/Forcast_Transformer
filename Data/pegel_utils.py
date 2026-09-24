
from _datetime import datetime, timedelta
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf



# Gibt Dataframe zurück mit Pegelständen nach Tagen.
def load_data(date_from, date_to=None):
    if date_to is None:
        date_to = datetime.now()
    current_dir = Path(__file__).parent

    csv_path = current_dir / 'pegel_daten.csv'

    df = pd.read_csv(csv_path, sep=';')
    df['time'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M')
    df = df[df['time'] >= date_from].sort_values('time')

    df = df[df['time'] <= date_to]

    df_daily = df.set_index('time').resample('2h')['value'].mean().interpolate(method='linear').reset_index()

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



def evaluate_and_plot_forecast(
    df,
    y_pred,
    time_col="time",
    value_col="value",
    show_train_days=200,
):
    """Berechnet Evaluation-Metriken (MSE, RMSE, MAE) und plottet Trainingsdaten,

    echte Testdaten sowie die Vorhersage.

    Parameters:
    -----------
    df : pd.DataFrame
        Das vollständige original DataFrame (enthält Training + Testdaten).
    y_pred : array-like or pd.Series
        Die vom Modell generierten Prognosewerte.
    time_col : str, optional
        Name der Zeitspalte im DataFrame (Standard: 'time').
    value_col : str, optional
        Name der Wertespalte im DataFrame (Standard: 'value').
    show_train_days : int, optional
        Anzahl der vergangenen Trainingstage im Plot zur besseren Übersicht
        (Standard: 200).

    Returns:
    --------
    dict
        Dictionary mit den berechneten Metriken {'MSE': ..., 'RMSE': ..., 'MAE':
        ...}
    """
    # Die Anzahl der vorhergesagten Tage leitet sich aus der Länge der Prognose ab
    forecast_days = len(y_pred)

    # Trennung in Training und Test basierend auf der Länge der Vorhersage
    train_data = df.iloc[:-forecast_days]
    test_data = df.iloc[-forecast_days:]

    y_true = test_data[value_col]
    test_time = test_data[time_col]

    # Metriken berechnen
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # Metriken in der Konsole ausgeben
    print("\n--- Prognosegüte ---")
    print(f"Vorhergesagte Tage: {forecast_days}")
    print(f"MSE  (Mean Squared Error):   {mse:.2f}")
    print(f"RMSE (Root Mean Sq. Error):  {rmse:.2f}")
    print(f"MAE  (Mean Absolute Error):  {mae:.2f}")

    # Plot erstellen
    plt.figure(figsize=(14, 7))

    # Trainingsdaten plotten (gekürzt auf show_train_days für bessere Lesbarkeit)
    train_plot = (
        train_data.iloc[-show_train_days:]
        if (show_train_days and show_train_days < len(train_data))
        else train_data
    )

    plt.plot(
        train_plot[time_col],
        train_plot[value_col],
        label=f"Trainingsdaten (letzte {len(train_plot)} Tage)",
        color="blue",
        alpha=0.6,
    )

    # Reale Testdaten plotten
    plt.plot(test_time, y_true, label="Echte Werte (Test Set)", color="orange")

    # Vorhersage plotten
    plt.plot(
        test_time,
        y_pred,
        label="Vorhersage",
        color="red",
        linestyle="--",
    )

    plt.title("Pegelstände: Reale Daten vs. Forecast")
    plt.xlabel("Datum")
    plt.ylabel("Pegelstand")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

    # Gibt die Werte zur optionalen Weiterverarbeitung im Code zurück
    return {"MSE": mse, "RMSE": rmse, "MAE": mae}


def analyze_pegel_data(df):
    # 1. Datenvorbereitung
    df = df.copy()
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

    df = df.sort_index()
    series = df['value'].dropna()

    # Extremwerte und deren Daten ermitteln
    min_val = series.min()
    min_date = series.idxmin().strftime('%d.%m.%Y')

    max_val = series.max()
    max_date = series.idxmax().strftime('%d.%m.%Y')

    # ==========================================
    # 2. Statistische Metriken berechnen
    # ==========================================
    print("--- Hydrologische & Statistische Metriken ---")
    print(f"Zeitraum: {series.index.min().strftime('%d.%m.%Y')} bis {series.index.max().strftime('%d.%m.%Y')}")
    print(f"Anzahl Messungen: {len(series)}\n")

    print("Zentralmaße & Streuung:")
    print(f"Mittelwert (Mean):    {series.mean():.2f}")
    print(f"Median (50% Quantil): {series.median():.2f}")
    print(f"Standardabweichung:   {series.std():.2f}\n")

    print("Extremwerte & Quantile:")
    print(f"Minimum:              {min_val:.2f}  (aufgetreten am {min_date})")
    print(f"Niedrigwasser (5%):   {series.quantile(0.05):.2f}")
    print(f"Hochwasser (95%):     {series.quantile(0.95):.2f}")
    print(f"Extremes HW (99%):    {series.quantile(0.99):.2f}")
    print(f"Maximum:              {max_val:.2f}  (aufgetreten am {max_date})\n")

    print("Verteilungsform:")
    print(f"Schiefe (Skewness):   {series.skew():.2f}")
    print(f"Kurtosis:             {series.kurtosis():.2f}")
    print("-" * 45)

    # ==========================================
    # 3. Visuelle Analyse (Plots)
    # ==========================================
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(15, 12))

    # Plot 1: Zeitverlauf mit gleitendem Durchschnitt (z.B. 30 Tage)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(series.index, series.values, label='Tageswert', color='lightblue', alpha=0.7)
    ax1.plot(series.index, series.rolling(window=30, center=True).mean(),
             label='30-Tage Durchschnitt', color='darkblue', linewidth=2)
    ax1.set_title('Pegelverlauf über die Zeit')
    ax1.set_ylabel('Pegelstand')
    ax1.legend()

    # Plot 2: Verteilung (Histogramm + Dichteschätzung)
    ax2 = plt.subplot(2, 2, 2)
    sns.histplot(series, bins=50, kde=True, ax=ax2, color='teal')
    ax2.axvline(series.mean(), color='red', linestyle='--', label=f'Mean: {series.mean():.1f}')
    ax2.axvline(series.median(), color='green', linestyle='-', label=f'Median: {series.median():.1f}')
    ax2.set_title('Verteilung der Pegelstände')
    ax2.set_xlabel('Pegelstand')
    ax2.legend()

    # Plot 3: Saisonalität (Boxplot pro Monat)
    ax3 = plt.subplot(2, 2, 3)
    df_monthly = pd.DataFrame({'value': series, 'month': series.index.month})
    sns.boxplot(x='month', y='value', data=df_monthly, ax=ax3, palette='Blues')
    ax3.set_title('Saisonalität: Pegelstände nach Monat')
    ax3.set_xlabel('Monat (1 = Jan, 12 = Dez)')
    ax3.set_ylabel('Pegelstand')

    # Plot 4: Autokorrelation (Gedächtnis des Flusses)
    ax4 = plt.subplot(2, 2, 4)
    # Zeigt bis zu 30 Tage in die Vergangenheit (lags=30)
    plot_acf(series, lags=30, ax=ax4, color='navy', alpha=0.05)
    ax4.set_title('Autokorrelation (ACF) - Trägheit des Systems')
    ax4.set_xlabel('Verzögerung in Tagen (Lag)')
    ax4.set_ylabel('Korrelation')

    plt.tight_layout()
    plt.show()