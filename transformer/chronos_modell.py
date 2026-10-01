import pandas as pd
import numpy as np
import os
import random
import torch
import matplotlib.pyplot as plt
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from lightning.pytorch.loggers import CSVLogger
from Data.pegel_utils import load_data

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tqdm import tqdm



# ---------------------------------------------------------
# 1. Daten laden und aufbereiten
# ---------------------------------------------------------
pegel_data = load_data('2000-01-01')
df = pd.DataFrame({
    'unique_id': ['pegel_1'] * len(pegel_data),
    'time': pd.to_datetime(pegel_data['time']),
    'value': pegel_data['value']
})


ts_dataframe = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="unique_id",
    timestamp_column="time"
)

print(ts_dataframe.head())

# ---------------------------------------------------------
# 2. Dynamischer 80% / 20% Split
# ---------------------------------------------------------
total_len = len(ts_dataframe)
test_size = int(0.20 * total_len)  # 20% für den finalen Test

# Training: Die ersten 80% der Daten
train_data = ts_dataframe.iloc[:-test_size]

# Test: Der gesamte Datensatz. Die Rolling-Window-Schleife
# startet später automatisch ab dem Index 'len(train_data)'.
test_data = ts_dataframe

print(f"Gesamtdaten: {total_len} Tage")
print(f"Training:    {len(train_data)} Tage (80%)")
print(f"Test-Bereich:{test_size} Tage (20%)")

print("\n--- Train Data Info ---")
print(train_data.info())

prediction_length = 30

csv_logger = CSVLogger(save_dir="ts_model_logs", name="deep_learning_logs")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)

# ---------------------------------------------------------
# 3. Predictor initialisieren
# ---------------------------------------------------------
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target="value",
    eval_metric="MSE",
    verbosity=4
    # known_covariates_names=["monat"]
)

# ---------------------------------------------------------
# 4. Gezielt NUR das Modell trainieren
# ---------------------------------------------------------
predictor.fit(
    train_data,
    hyperparameters={
        "Chronos2": {}
    },
    time_limit=None,
    random_seed=42,
    enable_ensemble=True
)

# ---------------------------------------------------------
# 5. Prognose auf Testdaten (Rolling Window)
# ---------------------------------------------------------
# Schrittgröße exakt wie in patch_tst.py auf 1 setzen
step = 30
max_start_idx = len(test_data) - prediction_length

print(f"\nStarte Rolling Window Evaluation (Schrittgröße: {step}) auf Originaldaten...")

metrics_list = []
y_true_all = []
y_pred_all = []

# Iteration über das Test-Set analog zu patch_tst.py
for start_idx in tqdm(range(len(train_data), max_start_idx + 1, step), desc="Evaluierung von Test-Set"):
    # Trainingsdaten-Schnitt für dieses Fenster
    current_train = test_data.iloc[:start_idx]
    # Die tatsächlichen Werte für den Vorhersagezeitraum
    current_test = test_data.iloc[start_idx: start_idx + prediction_length]

    if len(current_test) < prediction_length:
        break

    # Vorhersage für das aktuelle Fenster generieren
    predictions = predictor.predict(current_train)

    # Ausrichtung der echten und vorhergesagten Werte
    actuals = current_test['value'].values
    preds = predictions['mean'].values

    # Metriken für dieses spezifische Fenster berechnen
    mse = mean_squared_error(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actuals, preds)

    metrics_list.append({"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape})

    # Werte für den globalen Plot speichern
    y_true_all.extend(actuals)
    y_pred_all.extend(preds)

# ---------------------------------------------------------
# 6. Evaluierung & Plot (Gemitteltes Ergebnis & Einzelfenster)
# ---------------------------------------------------------
metrics_df = pd.DataFrame(metrics_list)

print("\n--- DURCHSCHNITTLICHE TESTDATEN METRIKEN (Rolling Window) ---")
print(f"Mean MSE:       {metrics_df['MSE'].mean():.4f}")
print(f"Mean RMSE:      {metrics_df['RMSE'].mean():.4f}")
print(f"Mean MAE:       {metrics_df['MAE'].mean():.4f}")
print(f"Mean MAPE:      {metrics_df['MAPE'].mean():.4f}")

print(f"Anzahl Fenster: {len(metrics_df)}")
print("-------------------------------------------------------------")

# 1. Arrays für den Plot vorbereiten
y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# 2. Scatter Plot erstellen (Observed vs. Chronos) identisch zu patch_tst.py
plt.figure(figsize=(6, 5), dpi=100)

plt.scatter(y_true_all, y_pred_all, alpha=0.5, color='#1f77b4', edgecolors='none', s=50)

# Perfekte Diagonale (y = x)
min_val = min(y_true_all.min(), y_pred_all.min())
max_val = max(y_true_all.max(), y_pred_all.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='1:1 Linie')

plt.xlabel('Observed', fontweight='bold', fontsize=12)
plt.ylabel('Chronos2', fontweight='bold', fontsize=12)
plt.title('Observed vs. Predicted', fontsize=12, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# erste Prognosefenster aus Test-Set plotten
# ---------------------------------------------------------

import matplotlib.pyplot as plt

# 1. Den ersten Vorhersagezeitraum definieren
# Startpunkt ist direkt nach den Trainingsdaten
start_idx = len(train_data)

# Historie für die Vorhersage (Kontext für das Modell)
current_train = test_data.iloc[:start_idx]

# Die echten Werte des ersten 30-Tage-Vorhersagezeitraums
first_test_window = test_data.iloc[start_idx : start_idx + prediction_length]

# 2. Vorhersage mit Chronos generieren
predictions = predictor.predict(current_train)

# 3. Zeitstempel und Werte extrahieren
timestamps = first_test_window.index.get_level_values('timestamp')
actual_values = first_test_window['value'].values
predicted_values = predictions['mean'].values

# 4. Plot erstellen
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(timestamps, actual_values, label='Tatsächliche Werte (Observed)', color='#1f77b4', marker='o', linewidth=2)
plt.plot(timestamps, predicted_values, label='Chronos Vorhersage (Predicted)', color='orange', linestyle='--', marker='s', linewidth=2)

plt.title('Erster Vorhersagezeitraum (30 Tage): Echte Werte vs. Chronos', fontsize=12, fontweight='bold')
plt.xlabel('Datum', fontweight='bold', fontsize=10)
plt.ylabel('Pegelstand', fontweight='bold', fontsize=10)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.gcf().autofmt_xdate()  # Schräge Datumsbeschriftung für bessere Lesbarkeit
plt.tight_layout()
plt.show()