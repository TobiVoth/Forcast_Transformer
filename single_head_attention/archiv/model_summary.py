from torchinfo import summary
import torch
from single_head_attention.archiv.singel_head_attention_modell import SimplePatchTST
from single_head_attention.archiv import hyperparameter

# Hyperparameter laden
lookback = hyperparameter.lookback
patch_len = hyperparameter.patch_len
pred_len = hyperparameter.pred_len
forecast_start_date = hyperparameter.forecast_start_date
d_model = hyperparameter.d_model
nhead = hyperparameter.nhead
use_data_from = hyperparameter.use_data_from
# Modell initialisieren und Gewichte laden
model = SimplePatchTST(
    lookback=lookback,
    patch_len=patch_len,
    d_model=d_model,
    nhead=nhead,
    pred_len=pred_len,
    eps=1e-5,


)
model.load_state_dict(torch.load('patchtst_model_weights.pth', weights_only=True))
model.eval()




# Erfordert ein Beispiel-Eingabetensor-Shape: (Batch_Size, Lookback)
summary(model, input_size=(32, lookback))