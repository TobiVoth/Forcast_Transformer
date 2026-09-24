import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from Data.pegel_utils import load_data
from single_head_attention.archiv import hyperparameter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_size_percent = hyperparameter.train_size_percent
lookback = hyperparameter.lookback
patch_len = hyperparameter.patch_len
pred_len = hyperparameter.pred_len
epochs = hyperparameter.epochs
forecast_start_date = hyperparameter.forecast_start_date
d_model = hyperparameter.d_model
nhead = hyperparameter.nhead
use_data_from = hyperparameter.use_data_from
stride = hyperparameter.stride

pegel_data = load_data(use_data_from)



# Fixieren der Seeds für Reproduzierbarkeit
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. PatchTST Modell-Architektur
# ==========================================
import torch
import torch.nn as nn

class SimplePatchTST(nn.Module):
    def __init__(self, lookback=336, patch_len=16, stride=8, d_model=32, nhead=1, pred_len=20, d_ff=128, dropout=0.3, eps=1e-5):
        super().__init__()
        self.lookback = lookback
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.eps = eps

        # Richtige Berechnung der Patch-Anzahl mit Stride
        self.num_patches = (lookback - patch_len) // stride + 1

        # 1. Patch-Projection (patch_len -> d_model)
        self.patch_proj = nn.Linear(patch_len, d_model)

        # Positional Encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # 2. Multi-Head Attention Sub-Layer
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)#, dropout=dropout
        self.norm1 = nn.LayerNorm(d_model)

        # 3. Feed-Forward Network (FFN) Sub-Layer
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU oder ReLU (PatchTST nutzt meist GELU)
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

        # 4. Forecast Head
        # 4. Forecast Head
        #self.head = nn.Linear(d_model, pred_len)
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    def forward(self, x, return_attn=False):
        batch_size = x.size(0)

        # RevIN Normalisierung
        mean = x.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)
        x_norm = (x - mean) / stdev

        # Unfold (Patching): Shape -> (Batch, Num_Patches, Patch_Len)
        x_patched = x_norm.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # Projection & Positional Embedding
        enc_in = self.patch_proj(x_patched) + self.pos_embed

        # --- Sub-Layer 1: Multi-Head Attention ---
        attn_out, attn_weights = self.attn(
            query=enc_in, key=enc_in, value=enc_in, need_weights=True, average_attn_weights=False
        )
        # Residual Connection & First LayerNorm
        x_attn = self.norm1(enc_in + attn_out)

        # --- Sub-Layer 2: Feed-Forward Network ---
        ffn_out = self.ffn(x_attn)
        # Residual Connection & Second LayerNorm
        enc_out = self.norm2(x_attn + ffn_out)

        # Flatten & Out Projection
        enc_out_flat = enc_out.reshape(batch_size, -1)
        out_norm = self.head(enc_out_flat)

        # Pooling & Out Projection
        #enc_out_pooled = enc_out.mean(dim=1)
        #out_norm = self.head(enc_out_pooled)

        # RevIN Denormalisierung
        out = out_norm * stdev + mean

        if return_attn:
            return out, attn_weights
        return out



if __name__ == '__main__':

    # ==========================================
    # 2. Datensynthese & Training Setup
    # ==========================================

    total_len = len(pegel_data)
    train_end = int(total_len * 0.80)
    val_end = int(total_len * 0.90)


    train_data = pegel_data.iloc[:train_end].copy()
    val_data = pegel_data.iloc[train_end:val_end].copy()
    test_data = pegel_data.iloc[val_end:].copy()

    print(f"Gesamte Tage/Datenpunkte: {total_len}")
    print(f"-> Training: {len(train_data)} (bis {train_data['time'].iloc[-1].date()})")
    print(f"-> Validation: {len(val_data)} (von {val_data['time'].iloc[0].date()} bis {val_data['time'].iloc[-1].date()})")
    print(f"-> Test: {len(test_data)} (ab {test_data['time'].iloc[0].date()})")

    train_values = train_data['value'].values
    val_values = val_data['value'].values
    test_values = test_data['value'].values

    def create_sliding_windows(data_array, lookback, pred_len):
        data_tensor = torch.tensor(data_array, dtype=torch.float32)
        windows = data_tensor.unfold(dimension=0, size=lookback + pred_len, step=1)
        X = windows[:, :lookback]
        y = windows[:, lookback:]
        return TensorDataset(X, y)


    # 3. Datasets & DataLoader für Train und Val erstellen
    train_dataset = create_sliding_windows(train_values, lookback, pred_len)
    val_dataset = create_sliding_windows(val_values, lookback, pred_len)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Modell, Optimizer & Loss initialisieren
    model = SimplePatchTST(lookback=lookback, patch_len=patch_len, stride=stride, d_model=d_model, nhead=nhead, pred_len=pred_len, eps=1e-5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # ==========================================
    # 3. Training Durchführen
    # ==========================================
    print("\nStarte Training...")

    train_losses = []
    val_losses = []
    patience = 15  # Wie viele Epochen ohne Verbesserung gewartet wird
    best_val_loss = float('inf')  # Startwert unendlich
    epochs_no_improve = 0
    best_model_path = 'best_model.pth'  # Pfad zum Speichern der Gewichte

    for epoch in range(epochs):
        # -- Trainings-Phase --
        model.train()
        total_train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -- Validierungs-Phase --
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for bx_val, by_val in val_loader:
                bx_val, by_val = bx_val.to(device), by_val.to(device)
                pred_val = model(bx_val)
                val_loss = criterion(pred_val, by_val)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        # --- Early Stopping & Model Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Speichere das aktuelle Modell, da es das bisher beste ist
            torch.save(model.state_dict(), best_model_path)

            # Optional: Print bei Verbesserung (kannst du auch auskommentieren)
            # print(f"Epoche {epoch+1}: Neues bestes Modell gespeichert (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        #if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Ohne Verbesserung: {epochs_no_improve}/{patience}")

        # Breche die Schleife ab, wenn Geduld am Ende ist
        if epochs_no_improve >= patience:
            print(f"\nEarly Stopping ausgelöst in Epoche {epoch + 1}!")
            break


    # ==========================================
    # 4. Loss Plotten
    # ==========================================

    actual_epochs = len(train_losses)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, actual_epochs + 1), train_losses, label='Train Loss (MSE)')
    plt.plot(range(1, actual_epochs + 1), val_losses, label='Val Loss (MSE)')
    plt.title('Training und Validation Loss')
    plt.xlabel('Epoche')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()


    #model.eval()
    #torch.save(model.state_dict(), 'patchtst_model_weights.pth')
    #print("Modell und Skalierer wurden erfolgreich gespeichert!")

