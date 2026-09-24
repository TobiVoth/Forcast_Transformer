from Data.pegel_utils import load_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# 1. Daten laden und vorbereiten
# ---------------------------------------------------------
pegel_data = load_data('2000-01-01')
df = pd.DataFrame({
    'unique_id': ['pegel_1'] * len(pegel_data),
    'ds': pd.to_datetime(pegel_data['time']),
    'y': pegel_data['value']
})

df = df.sort_values('ds').reset_index(drop=True)

# ---------------------------------------------------------
# 2. Dynamischer 70% / 10% / 20% Split
# ---------------------------------------------------------
total_len = len(df)
test_size = int(0.20 * total_len)  # 20% für den finalen Test
val_size  = int(0.10 * total_len)  # 10% für die Validierung während des Trainings
train_size = total_len - test_size - val_size # Die restlichen 70%

print(f"Gesamtdaten: {total_len} Tage")
print(f"Training:    {train_size} Tage (70%)")
print(f"Validierung: {val_size} Tage (10%)")
print(f"Test:        {test_size} Tage (20%)")

# Train/Val-Datensatz (80% der Daten: Training + Validierung)
train_val_df = df.iloc[:-test_size]

# Test-Datensatz (Die letzten 20%)
test_df = df.iloc[-test_size:]

# ---------------------------------------------------------
# 1. 30-Tage-Testfenster auswählen
# ---------------------------------------------------------
horizon = 120
test_30_df = test_df.iloc[:horizon].copy()

# ---------------------------------------------------------
# 2. Naive Vorhersage erstellen
# Letzter bekannter Wert aus train_val_df wird für 30 Tage fortgeschrieben
# ---------------------------------------------------------
last_known_value = train_val_df['y'].iloc[-1]
test_30_df['y_hat_naive'] = last_known_value

# ---------------------------------------------------------
# 3. RMSE berechnen
# ---------------------------------------------------------
rmse = root_mean_squared_error(test_30_df['y'], test_30_df['y_hat_naive'])
mae = mean_absolute_error(test_30_df['y'], test_30_df['y_hat_naive'])
mse = mean_squared_error(test_30_df['y'], test_30_df['y_hat_naive'])

print(f"MAE (Naive 30-Tage-Prognose): {mae:.4f}")
print(f"MSE (Naive 30-Tage-Prognose): {mse:.4f}")
print(f"RMSE (Naive 30-Tage-Prognose): {rmse:.4f}")

# ---------------------------------------------------------
# 4. Visualisierung
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))

# Kontext: Die letzten 60 Tage vor dem Split anzeigen
plt.plot(
    train_val_df['ds'].iloc[-60:],
    train_val_df['y'].iloc[-60:],
    label='Historie (Train/Val)',
    color='blue',
)

# Tatsächliche Werte der ersten 30 Test-Tage
plt.plot(
    test_30_df['ds'],
    test_30_df['y'],
    label='Tatsächliche Werte (Test)',
    color='black',
    linewidth=2,
)

# Naive Prognose
plt.plot(
    test_30_df['ds'],
    test_30_df['y_hat_naive'],
    label='Naive Prognose (30 Tage)',
    color='red',
    linestyle='--',
)

plt.title(f'30-Tage Naive Prognose vs. Reale Daten (RMSE: {rmse:.2f})')
plt.xlabel('Datum')
plt.ylabel('Pegelstand')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()