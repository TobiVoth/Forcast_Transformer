import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from singel_head_attention_modell import SimplePatchTST
from Data.pegel_utils import load_data
from single_head_attention.archiv import hyperparameter
from matplotlib import pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Hyperparameter laden (müssen zum gespeicherten Modell passen!)
lookback = hyperparameter.lookback
pred_len = hyperparameter.pred_len
patch_len = hyperparameter.patch_len
d_model = hyperparameter.d_model
nhead = hyperparameter.nhead
use_data_from = hyperparameter.use_data_from

# Daten laden und vorbereiten
pegel_data = load_data(use_data_from)
pegel_data['time'] = pd.to_datetime(pegel_data['time'])

total_len = len(pegel_data)
val_end = int(total_len * 0.90)

# Nur den Testdatensatz extrahieren
test_data = pegel_data.iloc[val_end:].copy()
test_values = test_data['value'].values

print(f"Gesamte Datenpunkte: {total_len}")
print(f"-> Testdaten: {len(test_data)} Punkte (ab {test_data['time'].iloc[0].date()})")

# Modell initialisieren und Gewichte laden
model = SimplePatchTST(
    lookback=lookback,
    patch_len=patch_len,
    d_model=d_model,
    nhead=nhead,
    pred_len=pred_len,
    eps=1e-5
).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
model.eval()

# Listen zum Sammeln der Metriken
mse_list = []
rmse_list = []
mae_list = []

# Prüfen, ob der Testdatensatz groß genug ist
if len(test_values) < lookback + pred_len:
    print("Fehler: Der Testdatensatz ist zu kurz für die gewählten Parameter.")
else:
    print("\nStarte Evaluierung über den Testdatensatz...")

    all_y_true = []
    all_y_pred = []

    # 1. Daten sammeln
    # Variablen für das erste Fenster bereitstellen
    first_y_true = None
    first_y_pred = None
    first_x_history = None

    with torch.no_grad():
        for i in range(len(test_values) - lookback - pred_len + 1):
            x_history = test_values[i: i + lookback]
            y_true = test_values[i + lookback: i + lookback + pred_len]

            x_tensor = torch.tensor(x_history, dtype=torch.float32).unsqueeze(0).to(device)
            y_pred = model(x_tensor).squeeze(0).cpu().numpy()

            # 1. Nur die Daten des ALLERERSTEN Fensters (Index 0) für den Plot speichern
            if i == 0:
                first_x_history = x_history
                first_y_true = y_true
                first_y_pred = y_pred

            # 2. Metriken für JEDES Fenster berechnen
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)

            mse_list.append(mse)
            mae_list.append(mae)
            rmse_list.append(rmse)

    # --- PLOT FÜR DAS ERSTE FENSTER ERSTELLEN ---
    plt.figure(figsize=(10, 5))

    # Zeitachsen definieren
    x_hist_range = range(0, lookback)
    x_pred_range = range(lookback, lookback + pred_len)

    # Historie (Input), echte Zukunft und Vorhersage plotten
    plt.plot(x_hist_range, first_x_history, label="Historie (x_history)", color="gray", marker="o")
    plt.plot(x_pred_range, first_y_true, label="Echte Zukunft (y_true)", color="blue", marker="o")
    plt.plot(x_pred_range, first_y_pred, label="Vorhersage (y_pred)", color="red", linestyle="--", marker="x")

    # Verbindungslinie vom letzten Historienpunkt zur Vorhersage
    plt.axvline(x=lookback - 1, color="black", linestyle=":", label="Vorhersage-Start")

    plt.title("Modell-Vorhersage vs. Echte Daten (Erstes Fenster: i = 0)")
    plt.xlabel("Zeitschritte im Fenster")
    plt.ylabel("Wert")
    plt.legend()
    plt.grid(True)
    plt.show()

    """
    
    with torch.no_grad():
        # Gleitendes Fenster Schritt für Schritt durch die Testdaten schieben
        # i ist der Startindex des aktuellen lookback-Fensters
        for i in range(len(test_values) - lookback - pred_len + 1):
            # 1. Historie und wahre Zukunft für diesen Schritt ausschneiden
            x_history = test_values[i: i + lookback]
            y_true = test_values[i + lookback: i + lookback + pred_len]

            # 2. In PyTorch-Tensor umwandeln und Batch-Dimension (unsqueeze) hinzufügen
            x_tensor = torch.tensor(x_history, dtype=torch.float32).unsqueeze(0).to(device)

            # 3. Vorhersage machen und Batch-Dimension entfernen
            y_pred = model(x_tensor).squeeze(0).cpu().numpy()
    """



    # Durchschnittliche, beste und schlechteste Metriken berechnen
    avg_mse = np.mean(mse_list)
    best_mse = np.min(mse_list)
    worst_mse = np.max(mse_list)

    avg_rmse = np.mean(rmse_list)
    best_rmse = np.min(rmse_list)
    worst_rmse = np.max(rmse_list)

    avg_mae = np.mean(mae_list)
    best_mae = np.min(mae_list)
    worst_mae = np.max(mae_list)

    print(f"Evaluierung abgeschlossen! Anzahl der Durchläufe: {len(mse_list)}")
    print(f"--------------------------------------------------")
    print(f"MSE  - Ø: {avg_mse:.4f} | Best: {best_mse:.4f} | Worst: {worst_mse:.4f}")
    print(f"RMSE - Ø: {avg_rmse:.4f} | Best: {best_rmse:.4f} | Worst: {worst_rmse:.4f}")
    print(f"MAE  - Ø: {avg_mae:.4f} | Best: {best_mae:.4f} | Worst: {worst_mae:.4f}")