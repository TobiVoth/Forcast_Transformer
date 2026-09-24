import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from single_head_attention.archiv import hyperparameter
from Data.pegel_utils import load_data
from singel_head_attention_modell import SimplePatchTST

# Hyperparameter laden
train_size_percent = hyperparameter.train_size_percent
lookback = hyperparameter.lookback
patch_len = hyperparameter.patch_len
pred_len = hyperparameter.pred_len
epochs = hyperparameter.epochs
forecast_start_date = hyperparameter.forecast_start_date
use_data_from = hyperparameter.use_data_from
stride = hyperparameter.stride
d_model = hyperparameter.d_model
nhead = hyperparameter.nhead

# Skalierer und Modell initialisieren & laden
model = SimplePatchTST(
    lookback=lookback,
    patch_len=patch_len,
    stride=stride,       # <--- STRIDE EINGEFÜGT!
    d_model=d_model,
    nhead=nhead,
    pred_len=pred_len,
    eps=1e-5,
)
model.load_state_dict(torch.load('best_model.pth', weights_only=True))
model.eval()

pegel_data = load_data(use_data_from)
pegel_data['time'] = pd.to_datetime(pegel_data['time'])

total_len = len(pegel_data)
val_end = int(total_len * 0.90)

# Nur den Testdatensatz extrahieren
test_data = pegel_data.iloc[val_end:].copy()
test_values = test_data['value'].values

with torch.no_grad():
    x_history = test_values[0: lookback]
    x_tensor = torch.tensor(x_history, dtype=torch.float32).unsqueeze(0)
    pred, attn_map = model(x_tensor, return_attn=True)

# 3. Attention Maps visualisieren (Für alle Heads)
num_patches = (lookback - patch_len) // stride + 1

# Da average_attn_weights=False ist, hat attn_map jetzt die Form (Batch, Heads, Patches, Patches)
attn_matrix = attn_map[0].numpy()

patch_labels = [f"P{i+1}" for i in range(num_patches)]

# Subplots nebeneinander erstellen (1 Zeile, nhead Spalten)
fig, axes = plt.subplots(1, nhead, figsize=(5 * nhead, 6))

# Falls nur 1 Head existiert, packen wir die axes in eine Liste, damit die Schleife funktioniert
if nhead == 1:
    axes = [axes]

for h in range(nhead):
    sns.heatmap(
        attn_matrix[h],
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=patch_labels,
        yticklabels=patch_labels,
        ax=axes[h]
    )
    axes[h].set_title(f"Head {h+1}", fontsize=14, fontweight='bold')
    axes[h].set_xlabel("Key Patch (Historie)", fontsize=10)
    axes[h].set_ylabel("Query Patch (Aktuell)" if h == 0 else "", fontsize=10)
    axes[h].tick_params(axis='x', rotation=45)

plt.suptitle(f"PatchTST Attention Maps (Startdatum: {forecast_start_date})", fontsize=16)
plt.tight_layout()
plt.show()