import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima.arima import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from Data.pegel_utils import load_data
from tqdm import tqdm

# from Data.pegel_utils import load_data, evaluate_and_plot_forecast

# ---------------------------------------------------------
# 1. Daten laden & vorbereiten
# ---------------------------------------------------------
pegel_data = load_data('2000-01-01')
horizon = [30,96,192,365]

for i in horizon:
    print('-' * 50)
    print(f'Evaluirung von Horizon {i}-Tage')
    print('-' * 50)
    # ---------------------------------------------------------
    # 2. Hyperparameter & Data Split
    # ---------------------------------------------------------
    train_size_percent = 0.8
    forecast_days = i

    train_auto = True
    p = 2
    d = 1
    q = 2

    split_idx = int(len(pegel_data) * train_size_percent)
    train_data = pegel_data.iloc[:split_idx].copy()
    test_data = pegel_data.iloc[split_idx:].copy()

    print(f"Gesamte Tage: {len(pegel_data)}")
    print(f"-> Davon Training: {len(train_data)} (bis {train_data['time'].iloc[-1].date()})")
    print(f"-> Davon Test:     {len(test_data)} (ab {test_data['time'].iloc[0].date()})")

    # ---------------------------------------------------------
    # 3. ARIMA Modell trainieren
    # ---------------------------------------------------------
    print("\nTrainiere ARIMA Modell (dies kann einen Moment dauern)...")
    train_data_short = train_data#.tail(2000)

    if train_auto is True:
        saison_periode = 6
        model = pm.auto_arima(
            train_data_short['value'],
            seasonal=False,
            # m=saison_periode,
            trace=True,
            suppress_warnings=True,
            stepwise=True,
            D=1
        )
        print("\nBeste Parameter (p, d, q):", model.order)
        print(model.summary())
    else:
        model = ARIMA(order=(p, d, q))
        model.fit(train_data_short['value'])
        print(model.summary())

    # ---------------------------------------------------------
    # 4. Rolling 30-Day Window Evaluation auf dem Test-Set
    # ---------------------------------------------------------
    print("\nStarte Rolling Window Evaluierung auf dem Test-Set...")

    HORIZON = forecast_days
    num_windows = len(test_data) - HORIZON + 1

    mse_list, rmse_list, mae_list, mape_list = [], [], [], []
    all_actuals, all_preds = [], []

    first_window_actuals = None
    first_window_preds = None
    first_window_dates = None

    # Schleife über das Test-Set
    for i in tqdm(range(num_windows), desc=f"Evaluierung ({HORIZON} Tage)"):
        actual_30_days = test_data['value'].iloc[i: i + HORIZON].values
        dates_30_days = test_data['time'].iloc[i: i + HORIZON].values

        # 30 Tage in die Zukunft vorhersagen basierend auf dem aktuellen Modellstatus
        pred_30_days = model.predict(n_periods=HORIZON)

        # Metriken berechnen
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

        # Erstes Fenster für den Plot speichern
        if i == 0:
            first_window_actuals = actual_30_days
            first_window_preds = pred_30_days
            first_window_dates = dates_30_days

        # MODELL UPDATE:
        # Den echten Wert von Tag 'i' dem Modell übergeben, damit das
        # nächste Fenster (i+1 bis i+30) diesen Datenpunkt als Historie nutzen kann.
        new_observation = test_data['value'].iloc[i: i + 1]
        model.update(new_observation)

    # ---------------------------------------------------------
    # 5. Durchschnittliche Metriken berechnen und ausgeben
    # ---------------------------------------------------------
    avg_mse = np.mean(mse_list)
    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)
    avg_mape = np.mean(mape_list)

    print("-" * 50)
    print(f"ARIMA Model ({HORIZON}-Day Rolling Forecast) - Test Set")
    print("-" * 50)
    print(f"Anzahl evaluierter {HORIZON}-Tage-Fenster: {num_windows}")
    print(f"Durchschnittlicher MSE:  {avg_mse:.4f}")
    print(f"Durchschnittlicher RMSE: {avg_rmse:.4f}")
    print(f"Durchschnittlicher MAE:  {avg_mae:.4f}")
    print(f"Durchschnittlicher MAPE: {avg_mape:.4%}")
    print("-" * 50)

    # ---------------------------------------------------------
    # 6. Visualisierung
    # ---------------------------------------------------------

    # --- Plot A: Das erste Prognosefenster mit Historie ---
    history_days = 30
    history_dates = train_data['time'].iloc[-history_days:].values
    history_actuals = train_data['value'].iloc[-history_days:].values

    combined_dates = np.concatenate([history_dates, first_window_dates])
    combined_actuals = np.concatenate([history_actuals, first_window_actuals])

    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(combined_dates, combined_actuals, marker='o', label='Echte Werte (Observed)', color='#1f77b4')
    plt.plot(first_window_dates, first_window_preds, linestyle='--', color='darkorange', linewidth=2,
             label=f'ARIMA Prognose ({HORIZON} Tage)')

    start_of_prediction = history_dates[-1]
    plt.axvline(x=start_of_prediction, color='gray', linestyle=':', linewidth=2, label='Start der Prognose')

    plt.title(f'{HORIZON}-Tage Prognosefenster (mit Historie)', fontsize=12, fontweight='bold')
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
    plt.ylabel('ARIMA Prediction', fontweight='bold', fontsize=12)
    plt.title(f'Observed vs. Predicted (Alle {HORIZON}-Tage Fenster)', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()