import torch
import matplotlib.pyplot as plt
import seaborn as sns

def predict_and_plot_attention(nf_model, df, layer_index=0):
    """
    Führt eine Prognose durch und plottet zuverlässig die Attention-Maps
    für jeden Head des angegebenen layer_index.
    """
    pytorch_model = nf_model.models[0]
    pytorch_model.eval()

    attention_maps = {}

    def hook_fn(module, input, output):
        # Wenn output ein Tuple ist: Das zweite Element sind bei nn.MultiheadAttention
        # gewöhnlich die reinen Softmax-Gewichte [Batch, 8, 8]
        if isinstance(output, tuple) and len(output) > 1:
            weights = output[1]
            if weights is not None:
                attention_maps['layer_output'] = weights.detach().cpu().numpy()
        elif isinstance(output, torch.Tensor) and output.ndim == 3:
            # Falls nur ein Tensor kommt, stellen wir sicher, dass es die 8x8 Matrix ist
            if output.shape[-1] == output.shape[-2]:  # Prüft ob Matrix quadratisch ist (8x8)
                attention_maps['layer_output'] = output.detach().cpu().numpy()

    try:
        # Zugriff auf die Multi-Head Attention des spezifizierten Encoder-Layers
        encoder_module = pytorch_model.model.backbone.encoder
        target_layer = encoder_module.layers[layer_index].self_attn

        # Erzwingen, dass das Modul die Attention Weights ausgibt
        if hasattr(target_layer, 'need_weights'):
            target_layer.need_weights = True

        handle = target_layer.register_forward_hook(hook_fn)
    except (AttributeError, IndexError) as e:
        print(f"Fehler: Encoder Layer {layer_index} konnte nicht adressiert werden ({e}).")
        return nf_model.predict(df=df)

    # Prognose ausführen (triggert den Hook)
    forecast_df = nf_model.predict(df=df)

    # Hook direkt danach wieder entfernen
    handle.remove()

    # Visualisierung aller Heads dieser Schicht
    if 'layer_output' in attention_maps and attention_maps['layer_output'] is not None:
        attn_data = attention_maps['layer_output']

        # Dimensionen anpassen: [Batch, Heads, Seq, Seq] -> [Heads, Seq, Seq]
        if len(attn_data.shape) == 4:
            first_batch_attn = attn_data[0]
        else:
            first_batch_attn = attn_data

        n_heads = first_batch_attn.shape[0]

        # Dynamische Figure-Größe je nach Anzahl der Heads
        fig, axes = plt.subplots(
            1, n_heads,
            figsize=(6 * n_heads, 5.5),
            squeeze=False
        )
        axes = axes.flatten()

        for head_idx in range(n_heads):
            sns.heatmap(
                first_batch_attn[head_idx],
                cmap='viridis',
                ax=axes[head_idx],
                cbar=True,
                square=True
            )
            axes[head_idx].set_title(f'Head {head_idx + 1}', fontsize=12)
            axes[head_idx].set_xlabel('Key (Patch Index)')
            axes[head_idx].set_ylabel('Query (Patch Index)')

        # Titel setzen
        fig.suptitle(f'PatchTST Attention Maps – Encoder Layer {layer_index}', fontsize=16)

        # Reserviert explizit 15% Platz am oberen Rand (top=0.85) für die Überschrift
        # und verhindert, dass Titel oder Achsen außerhalb des Bildes liegen
        plt.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.92, wspace=0.4)

        plt.show()
    else:
        print(f"Hinweis: Für Layer {layer_index} wurden keine Attention-Gewichte erfasst.")

    return forecast_df