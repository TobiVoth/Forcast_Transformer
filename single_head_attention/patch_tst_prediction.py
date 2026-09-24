import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from neuralforecast import NeuralForecast
from pytorch_lightning.loggers import CSVLogger
from Data.pegel_utils import load_data
import glob
import os
from attention_utils import predict_and_plot_attention
from neuralforecast.auto import AutoPatchTST
from neuralforecast.losses.pytorch import MSE
from ray import tune
import ray

# ---------------------------------------------------------
# Ray mit neuem, kurzem Verzeichnis initialisieren (gegen Windows MAX_PATH Fehler)
# ---------------------------------------------------------
# Falls noch eine alte Ray-Session läuft, beenden wir sie zuerst
if ray.is_initialized():
    ray.shutdown()

# Kurzen Pfad auf C: erstellen und Ray dort starten lassen
ray_temp_dir = "C:/ray_tmp"
os.makedirs(ray_temp_dir, exist_ok=True)

ray.init(_temp_dir=ray_temp_dir, ignore_reinit_error=True)
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
test_size = int(0.20 * total_len)
val_size = int(0.10 * total_len)
train_size = total_len - test_size - val_size

print(f"Gesamtdaten: {total_len} Tage")
print(f"Training:    {train_size} Tage (70%)")
print(f"Validierung: {val_size} Tage (10%)")
print(f"Test:        {test_size} Tage (20%)")

train_val_df = df.iloc[:-test_size]
test_df = df.iloc[-test_size:]

# ---------------------------------------------------------
# 3. AUTO Training und Tuning konfigurieren
# ---------------------------------------------------------
csv_logger = CSVLogger(save_dir="my_logs", name="autopatchtst_pegel")

# Vorhersagehorizont konstant auf 30 Tage gesetzt
h_horizon = 30

patchtst_config = {
    "input_size": tune.choice([120, 256, 512, 1024]),
    "patch_len": tune.choice([8, 16, 32]),
    "stride": tune.choice([8, 16]),
    "encoder_layers": tune.choice([1, 2, 3, 5]),
    "n_heads": tune.choice([1, 2]),
    "hidden_size": tune.choice([32, 64, 128]),
    "dropout": tune.uniform(0.1, 0.3),
    "learning_rate": tune.loguniform(1e-4, 1e-2),
    "max_steps": 1000,
    "val_check_steps": 20,
    "batch_size": 32,
    "early_stop_patience_steps": 5
}

model = AutoPatchTST(
    h=h_horizon,
    config=patchtst_config,
    loss=MSE(),        # Trainings-Loss auf MSE setzen
    valid_loss=MSE(),
    num_samples=10
)

nf = NeuralForecast(models=[model], freq='D')

# ---------------------------------------------------------
# 4. Training (inkl. Validierung)
# ---------------------------------------------------------
print("\nStarte Hyperparameter-Tuning (kann etwas dauern)...")
nf.fit(df=train_val_df, val_size=val_size)
# Nach dem Training Ray wieder sauber beenden


# ---------------------------------------------------------
# 4.1 Auswertung aller getesteten Modelle (Ray Tune Results)
# ---------------------------------------------------------
# Die Ergebnistabelle aus dem AutoPatchTST Modell abrufen
tune_results = nf.models[0].results
# ---------------------------------------------------------
# 4.1 Auswertung aller getesteten Modelle (Ray Tune Results)
# ---------------------------------------------------------
# Ray ResultGrid in ein Pandas DataFrame umwandeln
results_grid = nf.models[0].results
tune_results = results_grid.get_dataframe()

print("\n=============================================================")
print("                   AUSWERTUNG HYPERPARAMETER-TUNING           ")
print("=============================================================")
print("Auswahlkriterium: Kleinster Validation Loss (MSE auf Validierungsset)")
print(f"Anzahl getesteter Modelle: {len(tune_results)}")

# Relevanteste Spalten herausfiltern
config_cols = [col for col in tune_results.columns if col.startswith('config/')]

# Sicherstellen, dass 'loss' oder 'valid_loss' vorhanden ist
loss_col = 'loss' if 'loss' in tune_results.columns else 'valid_loss'

summary_cols = ['trial_id', loss_col] + config_cols

# Sortieren nach dem Validation Loss (niedrigster Wert zuerst)
summary_df = tune_results[summary_cols].sort_values(loss_col).reset_index(drop=True)

# Unnötiges Präfix 'config/' für eine saubere Anzeige entfernen
summary_df.columns = [c.replace('config/', '') for c in summary_df.columns]

print("\n--- Rangliste aller 10 getesteten Modelle ---")
print(summary_df.to_string(index=True))



ray.shutdown()

# ---------------------------------------------------------
# 5. Modell speichern
# ---------------------------------------------------------
nf.save(path='./checkpoints/pegel_autopatchtst_model/', overwrite=True)
print("Modell erfolgreich gespeichert!")

# ---------------------------------------------------------
# 6. Einmalige Prognose inkl. Attention Plot
# ---------------------------------------------------------
print("\nErstelle Basis-Prognose und visualisiere Attention...")
forecast_df = predict_and_plot_attention(nf_model=nf, df=train_val_df, layer_index=0)

# WICHTIG: Spalte heißt jetzt 'AutoPatchTST'
results_df = pd.merge(test_df[['ds', 'y']], forecast_df[['ds', 'AutoPatchTST']], on='ds')
results_df.rename(columns={'y': 'Echte_Werte', 'AutoPatchTST': 'Prognose'}, inplace=True)

# ---------------------------------------------------------
# 7. Rolling Window Evaluierung auf Testdaten
# ---------------------------------------------------------
print("\nErstelle rollierende Prognose für Test-Zeitraum...")

# Wir nutzen das oben definierte 'h_horizon' (30)
metrics_list = []
all_forecasts = []

for i in range(0, len(test_df), h_horizon):
    if i + h_horizon > len(test_df):
        break

    current_history = pd.concat([train_val_df, test_df.iloc[:i]])
    current_true = test_df.iloc[i: i + h_horizon]

    current_pred_df = nf.predict(df=current_history)
    current_pred = current_pred_df.iloc[-h_horizon:]

    # WICHTIG: Spalte heißt 'AutoPatchTST'
    mse = mean_squared_error(current_true['y'], current_pred['AutoPatchTST'])
    mae = mean_absolute_error(current_true['y'], current_pred['AutoPatchTST'])
    rmse = np.sqrt(mse)

    metrics_list.append({'MSE': mse, 'RMSE': rmse, 'MAE': mae})

    all_forecasts.append({
        'history': current_history.tail(10),
        'true': current_true,
        'pred': current_pred
    })

# ---------------------------------------------------------
# 8. Evaluierung & Plot
# ---------------------------------------------------------
if metrics_list:
    metrics_df = pd.DataFrame(metrics_list)

    print("\n--- DURCHSCHNITTLICHE TESTDATEN METRIKEN (Rolling Window) ---")
    print(f"Mean MSE:       {metrics_df['MSE'].mean():.4f}")
    print(f"Mean RMSE:      {metrics_df['RMSE'].mean():.4f}")
    print(f"Mean MAE:       {metrics_df['MAE'].mean():.4f}")
    print(f"Anzahl Fenster: {len(metrics_df)}")
    print("-------------------------------------------------------------")

    plot_data = all_forecasts[0]
    hist_df = plot_data['history']
    true_df = plot_data['true']
    pred_df = plot_data['pred']

    plt.figure(figsize=(12, 6))
    plt.plot(hist_df['ds'], hist_df['y'], label='Historie (10 Tage)', color='gray', marker='o')
    plt.plot(true_df['ds'], true_df['y'], label='Echte Werte', color='blue', marker='o')

    # WICHTIG: Spalte heißt 'AutoPatchTST'
    plt.plot(pred_df['ds'], pred_df['AutoPatchTST'], label='Prognose (AutoPatchTST)', color='red', linestyle='--',
             marker='x')

    plt.axvline(x=true_df['ds'].iloc[0], color='black', linestyle=':', label='Start Prognose-Fenster')
    plt.title(f'Pegelstand Einzel-Prognose (Horizont: {h_horizon} Tage)')
    plt.xlabel('Datum')
    plt.ylabel('Pegelstand')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Warnung: Test-Datensatz war zu kurz für ein vollständiges Rolling Window.")