"""from Data.pegel_utils import load_data, plot_pegel_data, analyze_pegel_data
import pandas as pd

#Läd Pegeldaten von bestimmtem Zeitpunkt
pegel_data = load_data('2026-09-01', '2026-09-22')

print(pegel_data.head())
print(pegel_data.describe())

#Plotet daten ab gewähltem Datum
#optional kann ein plot_to Datum mitgegeben werden, bis wann die Daten geplottet werden sollen
plot_from = '2026-09-01'
plot_pegel_data(pegel_data, plot_from)



analyze_pegel_data(pegel_data)

"""
"""
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Nutze Device: {device}")

import torch

print("PyTorch Version:", torch.__version__)
print("CUDA verfügbar?:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Grafikkarte:", torch.cuda.get_device_name(0))
else:
    print("CUDA Version in PyTorch:", torch.version.cuda)
"""


"""
import matplotlib.pyplot as plt
from Data.pegel_utils import load_data
import single_head_attention.hyperparameter as hyperparameter
use_data_from = hyperparameter.use_data_from


pegel_data = load_data(use_data_from)



total_len = len(pegel_data)
train_end = int(total_len * 0.80)
val_end = int(total_len * 0.90)

train_data = pegel_data.iloc[:train_end].copy()
val_data = pegel_data.iloc[train_end:val_end].copy()
test_data = pegel_data.iloc[val_end:].copy()







plt.figure(figsize=(14, 6))

# Plot für jedes Set (X-Achse: Zeit, Y-Achse: Pegelwert)
plt.plot(train_data['time'], train_data['value'], label='Train Data', color='blue', alpha=0.8)
plt.plot(val_data['time'], val_data['value'], label='Validation Data', color='orange', alpha=0.8)
plt.plot(test_data['time'], test_data['value'], label='Test Data', color='green', alpha=0.8)

# Formatierung des Plots
plt.title('Train-, Validation- und Test-Split der Pegeldaten')
plt.xlabel('Datum')
plt.ylabel('Pegelstand')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Plot anzeigen
plt.show()

"""


from neuralforecast import NeuralForecast

# 1. Das gespeicherte Modell aus dem Verzeichnis laden
loaded_nf = NeuralForecast.load(path='./single_head_attention/checkpoints/pegel_autopatchtst_model/')

# 2. Zugriff auf das erste (und einzige) Modell
model = loaded_nf.models[0]

# 3. Konfiguration anzeigen
print("--- Geladene Modell-Konfiguration ---")

# Falls es als AutoPatchTST geladen wird, stehen die Parameter direkt am Modell-Objekt:
attrs = ['h', 'input_size', 'patch_len', 'stride', 'encoder_layers', 'n_heads', 'hidden_size', 'dropout', 'learning_rate']

for attr in attrs:
    if hasattr(model, attr):
        print(f"{attr:<15}: {getattr(model, attr)}")
    elif hasattr(model, 'hparams') and attr in model.hparams:
        print(f"{attr:<15}: {model.hparams[attr]}")