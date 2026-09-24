import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from pytorch_lightning.loggers import CSVLogger
from Data.pegel_utils import load_data
import glob
import os
from attention_utils import predict_and_plot_attention
from neuralforecast.losses.pytorch import MSE

from neuralforecast.auto import AutoPatchTST
from ray import tune

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
# 3. Modell konfigurieren
# ---------------------------------------------------------
csv_logger = CSVLogger(save_dir="my_logs", name="patchtst_pegel")

model = PatchTST(
    h=30,
    input_size=120,
    patch_len=16,
    stride=16,
    encoder_layers=3,
    n_heads=1,
    hidden_size=64,
    max_steps=1000,
    val_check_steps=20,
    early_stop_patience_steps=5,
    learning_rate=0.000335,
    dropout=0.140691,
    batch_size=32,
    logger=csv_logger,
    loss=MSE(),        # Trainings-Loss auf MSE setzen
    valid_loss=MSE(),
)

nf = NeuralForecast(models=[model], freq='2h')





# ---------------------------------------------------------
# 4. Training (inkl. Validierung)
# ---------------------------------------------------------
# NeuralForecast nutzt intern die letzten `val_size` Zeilen von train_val_df zur Validierung
print("\nStarte Training...")
nf.fit(df=train_val_df, val_size=val_size)


# ---------------------------------------------------------
# 4.1 Loss-Funktion plotten
# ---------------------------------------------------------
# Wir suchen jetzt in dem explizit definierten Verzeichnis
log_files = glob.glob("my_logs/patchtst_pegel/**/metrics.csv", recursive=True)

if log_files:
    latest_log = max(log_files, key=os.path.getmtime)
    print(f"Lese Logs aus: {latest_log}")

    metrics_df = pd.read_csv(latest_log)

    plt.figure(figsize=(10, 5))

    # 1. Trainings-Loss plotten (Nutzt nun primär 'train_loss_epoch')
    if 'train_loss_epoch' in metrics_df.columns:
        # Extrahiere nur Zeilen, die einen Wert für train_loss_epoch haben
        train_data = metrics_df[['step', 'train_loss_epoch']].dropna()
        if not train_data.empty:
            plt.plot(train_data['step'], train_data['train_loss_epoch'],
                     label='Train Loss (Epoch)', color='blue', alpha=0.8, marker='.')
    # Fallback, falls zukünftige Logs doch 'train_loss_step' verwenden
    elif 'train_loss_step' in metrics_df.columns:
        train_data = metrics_df[['step', 'train_loss_step']].dropna()
        if not train_data.empty:
            plt.plot(train_data['step'], train_data['train_loss_step'],
                     label='Train Loss (Step)', color='blue', alpha=0.8, marker='.')

    # 2. Validierungs-Loss plotten (Nutzt die Spalte 'valid_loss' oder 'ptl/val_loss')
    val_col = 'valid_loss' if 'valid_loss' in metrics_df.columns else 'ptl/val_loss'

    if val_col in metrics_df.columns:
        # Extrahiere nur Zeilen, die einen Wert für valid_loss haben
        val_data = metrics_df[['step', val_col]].dropna()
        if not val_data.empty:
            plt.plot(val_data['step'], val_data[val_col],
                     label='Validation Loss', color='orange', linewidth=2, marker='o')

    plt.xlabel('Step')
    plt.ylabel('Loss (MSE)')
    plt.title('PatchTST: Training vs. Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Finale Werte ausgeben
    if 'train_loss_epoch' in metrics_df.columns and not train_data.empty:
        print(f"Finaler Train Loss: {train_data['train_loss_epoch'].iloc[-1]:.4f}")
    elif 'train_loss_step' in metrics_df.columns and not train_data.empty:
        print(f"Finaler Train Loss: {train_data['train_loss_step'].iloc[-1]:.4f}")

    if val_col in metrics_df.columns and not val_data.empty:
        print(f"Finaler Val Loss:   {val_data[val_col].iloc[-1]:.4f}")
else:
    print("Fehler: Keine metrics.csv gefunden.")

# ---------------------------------------------------------
# 5. Modell speichern
# ---------------------------------------------------------
nf.save(path='./checkpoints/pegel_patchtst_model/', overwrite=True)
print("Modell erfolgreich gespeichert!")





# ---------------------------------------------------------
# 5. Prognose auf Testdaten (Rolling Window)
# ---------------------------------------------------------
best_model = nf.models[0]

# Liest die tatsächliche Anzahl der Encoder-Schichten aus dem Backbone
try:
    num_layers = len(best_model.model.backbone.encoder.layers)
except AttributeError:
    num_layers = getattr(best_model, 'encoder_layers', 1)

print(f"\nVisualisiere Attention Maps für alle {num_layers} Encoder-Schichten...")

forecast_results = []
for layer_idx in range(num_layers):
    print(f"--> Rendere Encoder Layer {layer_idx}...")

    # Funktionsaufruf zeigt den Plot für Layer 'layer_idx' mit allen Heads nebeneinander an
    forecast_df = predict_and_plot_attention(
        nf_model=nf,
        df=train_val_df,
        layer_index=layer_idx
    )
    forecast_results.append(forecast_df)


# Danach geht Ihr Code ganz normal weiter:
results_df = pd.merge(test_df[['ds', 'y']], forecast_df[['ds', 'PatchTST']], on='ds')
results_df.rename(columns={'y': 'Echte_Werte', 'PatchTST': 'Prognose'}, inplace=True)
print("\nErstelle rollierende Prognose für Test-Zeitraum...")


# Der tatsächliche Vorhersagehorizont des Modells (z.B. 7 Tage)
# Muss mit dem 'h' aus deiner PatchTST-Konfiguration übereinstimmen!
h = model.h

metrics_list = []
all_forecasts = []

# Wir iterieren in Schritten von 'h' durch das Test-Set
for i in range(0, len(test_df), h):
    # Abbrechen, wenn nicht mehr genug echte Testdaten für ein ganzes h-Fenster übrig sind
    if i + h > len(test_df):
        break

    # Die Historie baut sich stückweise auf:
    # Ursprüngliche Trainingsdaten + die echten Testdaten bis zum aktuellen Schritt 'i'
    current_history = pd.concat([train_val_df, test_df.iloc[:i]])

    # Die wahren Werte, die wir in diesem Schritt vorhersagen wollen
    current_true = test_df.iloc[i: i + h]

    # Vorhersage generieren (NeuralForecast nutzt intern die letzten 'input_size' Tage der Historie)
    current_pred_df = nf.predict(df=current_history)
    current_pred = current_pred_df.iloc[-h:]  # Nur die neu vorhergesagten 'h' Tage nehmen

    # Metriken für dieses spezifische Fenster berechnen
    mse = mean_squared_error(current_true['y'], current_pred['PatchTST'])
    mae = mean_absolute_error(current_true['y'], current_pred['PatchTST'])
    rmse = np.sqrt(mse)

    metrics_list.append({'MSE': mse, 'RMSE': rmse, 'MAE': mae})

    # Daten für einen späteren Plot speichern
    all_forecasts.append({
        'history': current_history.tail(120),  # Genau 10 Tage Historie
        'true': current_true,
        'pred': current_pred
    })

# ---------------------------------------------------------
# 6. Evaluierung & Plot (Gemitteltes Ergebnis & Einzelfenster)
# ---------------------------------------------------------
metrics_df = pd.DataFrame(metrics_list)

print("\n--- DURCHSCHNITTLICHE TESTDATEN METRIKEN (Rolling Window) ---")
print(f"Mean MSE:       {metrics_df['MSE'].mean():.4f}")
print(f"Mean RMSE:      {metrics_df['RMSE'].mean():.4f}")
print(f"Mean MAE:       {metrics_df['MAE'].mean():.4f}")
print(f"Anzahl Fenster: {len(metrics_df)}")
print("-------------------------------------------------------------")

# Wir wählen das ERSTE Vorhersagefenster für den Plot aus.
# (Du kannst den Index all_forecasts[0] ändern, z.B. auf [-1] für das letzte Fenster)
plot_data = all_forecasts[0]
hist_df = plot_data['history'].reset_index(drop=True)
true_df = plot_data['true'].reset_index(drop=True)
pred_df = plot_data['pred'].reset_index(drop=True)

plt.figure(figsize=(14, 6))

# 1. Daten plotten (Spaltenname 'PatchTST' korrekt verwenden!)
plt.plot(hist_df['ds'], hist_df['y'], label='Historie (120 Tage)', color='gray', marker='o', alpha=0.7)
plt.plot(true_df['ds'], true_df['y'], label='Echte Werte', color='blue', marker='o')
plt.plot(pred_df['ds'], pred_df['PatchTST'], label='Prognose (PatchTST)',
         color='red', linestyle='--', marker='x')

# 2. Start der Prognose markieren
start_pred_date = true_df['ds'].iloc[0]
plt.axvline(x=start_pred_date, color='black', linestyle='-', linewidth=2, label='Start Prognose-Fenster')

# 3. Patch-Grenzen als gestrichelte Linien einzeichnen
best_model = nf.models[0]
patch_len = getattr(best_model, 'patch_len', 16)

# Nach links in die Historie zeichnen
first_hist_date = hist_df['ds'].iloc[0]
curr_date = start_pred_date - pd.Timedelta(days=patch_len)

patch_count = 1
while curr_date >= first_hist_date:
    plt.axvline(x=curr_date, color='orange', linestyle='--', alpha=0.7,
                label='Patch-Grenze' if patch_count == 1 else "")
    curr_date -= pd.Timedelta(days=patch_len)
    patch_count += 1

# Nach rechts in den Vorhersagehorizont zeichnen
last_pred_date = true_df['ds'].iloc[-1]
curr_date = start_pred_date + pd.Timedelta(days=patch_len)

while curr_date <= last_pred_date:
    plt.axvline(x=curr_date, color='orange', linestyle='--', alpha=0.7)
    curr_date += pd.Timedelta(days=patch_len)

# ---------------------------------------------------------
plt.title(f'Pegelstand Einzel-Prognose (Horizont: {h} Tage, Patch-Länge: {patch_len} Tage)')
plt.xlabel('Datum')
plt.ylabel('Pegelstand')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()