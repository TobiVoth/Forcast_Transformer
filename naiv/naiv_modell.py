import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from Data.pegel_utils import load_data

pegel_data = load_data('2000-01-01')
# ---------------------------------------------------------
# 1. Daten laden & vorbereiten (Dein Code)
# ---------------------------------------------------------


def execute_naive(pegel_data, horizon=[30, 40] ):
    df = pd.DataFrame({
        'unique_id': ['pegel_1'] * len(pegel_data),
        'ds': pd.to_datetime(pegel_data['time']),
        'y': pegel_data['value']
    })

    df = df.sort_values('ds').reset_index(drop=True)

# ---------------------------------------------------------
# 2. Hyperparameter & Data Split
# ---------------------------------------------------------
    total_len = len(df)
    test_size = int(0.20 * total_len)
    val_size  = int(0.10 * total_len)
    train_size = total_len - test_size - val_size

    print(f"Gesamtdaten: {total_len} Tage")
    print(f"Training:    {train_size} Tage (70%)")
    print(f"Validierung: {val_size} Tage (10%)")
    print(f"Test:        {test_size} Tage (20%)")

    train_val_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()

# ---------------------------------------------------------
# 3. Rolling 30-Day Window Evaluation (Naive Model)
# ---------------------------------------------------------
    for i in horizon:

        print('-'* 50)
        print(f'Evaluirung von Horizon {i}-Tage')
        print('-'* 50)

        HORIZON = i
        mse_list = []
        mae_list = []
        rmse_list = []
        mape_list = []



        all_actuals = []
        all_preds = []

        first_window_actuals = None
        first_window_preds = None
        first_window_dates = None


        num_windows = len(test_df) - HORIZON + 1

        for i in range(num_windows):
            actual_30_days = test_df['y'].iloc[i: i + HORIZON].values
            dates_30_days = test_df['ds'].iloc[i: i + HORIZON].values

            if i == 0:
                last_known_value = train_val_df['y'].iloc[-1]
            else:
                last_known_value = test_df['y'].iloc[i - 1]

            pred_30_days = np.full(HORIZON, last_known_value)

            window_mse = mean_squared_error(actual_30_days, pred_30_days)
            window_rmse = np.sqrt(window_mse)
            window_mae = mean_absolute_error(actual_30_days, pred_30_days)
            window_mape = mean_absolute_percentage_error(actual_30_days, pred_30_days)

            mse_list.append(window_mse)
            rmse_list.append(window_rmse)
            mae_list.append(window_mae)
            mape_list.append(window_mape)

            all_actuals.extend(actual_30_days)
            all_preds.extend(pred_30_days)

            if i == 0:
                first_window_actuals = actual_30_days
                first_window_preds = pred_30_days
                first_window_dates = dates_30_days

# ---------------------------------------------------------
# 4. Durchschnittliche Metriken berechnen
# ---------------------------------------------------------
        avg_mse = np.mean(mse_list)
        avg_rmse = np.mean(rmse_list)
        avg_mae = np.mean(mae_list)
        avg_mape = np.mean(mape_list)

        print("-" * 50)
        print(f"Naive Model ({HORIZON}-Days Rolling Forecast) - Test Set")
        print("-" * 50)
        print(f"Anzahl evaluierter Fenster: {num_windows}")
        print(f"Durchschnittlicher MSE:  {avg_mse:.4f}")
        print(f"Durchschnittlicher RMSE: {avg_rmse:.4f}")
        print(f"Durchschnittlicher MAE:  {avg_mae:.4f}")
        print(f"Durchschnittlicher MAPE: {avg_mape:.4%}") # Ausgabe in Prozent
        print("-" * 50)

# ---------------------------------------------------------
# 5. Visualisierung
# ---------------------------------------------------------

        # --- Plot A: Das erste Prognosefenster mit Historie ---
        history_days = 14
        # Hole die letzten 14 Tage aus dem train_val_df
        history_dates = train_val_df['ds'].iloc[-history_days:].values
        history_actuals = train_val_df['y'].iloc[-history_days:].values

        # Verbinde Historie und die echten Werte der Prognose für eine durchgehende Linie
        combined_dates = np.concatenate([history_dates, first_window_dates])
        combined_actuals = np.concatenate([history_actuals, first_window_actuals])

        plt.figure(figsize=(10, 5), dpi=100)
        # Linie für alle echten Werte (Vergangenheit + Zukunft)
        plt.plot(combined_dates, combined_actuals, marker='o', label='Echte Werte (Observed)', color='#1f77b4')

        # Linie für die Naive Prognose (nur in den 30 Tagen der Zukunft)
        plt.plot(first_window_dates, first_window_preds, linestyle='--', color='red', linewidth=2,
                 label=f'Naive Prognose ({HORIZON} Tage konstant)')

        # Eine vertikale Linie, um das "Jetzt" (Start der Prognose) zu markieren
        start_of_prediction = history_dates[-1]
        plt.axvline(x=start_of_prediction, color='gray', linestyle=':', linewidth=2, label='Start der Prognose')

        plt.title(f'Erstes {HORIZON}-Tage Prognosefenster', fontsize=12, fontweight='bold')
        plt.xlabel('Datum', fontweight='bold', fontsize=10)
        plt.ylabel('Pegelstand', fontweight='bold', fontsize=10)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

        # --- Plot B: Scatter-Plot (Alle Fenster aggregiert) ---
        plt.figure(figsize=(6, 5), dpi=100)
        plt.scatter(all_actuals, all_preds, alpha=0.05, color='#1f77b4', edgecolors='none', s=30)

        min_val = min(min(all_actuals), min(all_preds))
        max_val = max(max(all_actuals), max(all_preds))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='1:1 Linie')

        plt.xlabel('Observed', fontweight='bold', fontsize=12)
        plt.ylabel('Naive Prediction', fontweight='bold', fontsize=12)
        plt.title(f'Observed vs. Predicted (Alle {HORIZON}-Tage Fenster)', fontsize=12, fontweight='bold')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()



execute_naive(pegel_data, horizon=[30,96,192,365])