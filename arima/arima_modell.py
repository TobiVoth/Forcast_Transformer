import pandas as pd
import pmdarima as pm
from Data.pegel_utils import load_data, evaluate_and_plot_forecast
from pmdarima.arima import ARIMA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Hyperparameter
train_size_percent = 0.8
forecast_days = 30

# ARIMA
train_auto = True
p = 2  # Autoregressiv (Lags der Vergangenheit)
d = 1  # Differenzierung (Trend entfernen)
q = 2  # Moving Average (Fehlerkorrektur)



pegel_data = load_data('2000-01-01')


split_idx = int(len(pegel_data) * train_size_percent)

train_data = pegel_data.iloc[:split_idx]
test_data = pegel_data.iloc[split_idx:]

print(f"Gesamte Tage: {len(pegel_data)}")
print(f"-> Davon Training: {len(train_data)} (bis {train_data['time'].iloc[-1].date()})")
print(f"-> Davon Test: {len(test_data)} (ab {test_data['time'].iloc[0].date()})")



if train_auto is True:
    model = pm.auto_arima(
        train_data['value'],
        seasonal=False,       # Setze dies auf True, wenn du Saisonalität erwartest
        #m=6,               # Falls seasonal=True: m angeben (z.B. 12 für Monate)
        trace=True,           # Zeigt den Suchverlauf
        suppress_warnings=True,
        stepwise=True         # Macht die Suche deutlich schneller
    )

    # 4. Gefundene Parameter ausgeben
    print("\nBeste Parameter (p, d, q):", model.order)
    print(model.summary())
else:
    model = ARIMA(order=(p, d, q))
    model.fit(train_data['value'])
    print(model.summary())

forecast_values = model.predict(n_periods=forecast_days)

y_true = test_data['value'].iloc[:forecast_days]
y_pred = forecast_values

test_time = test_data['time'].iloc[:forecast_days]
forecast_values = model.predict(n_periods=forecast_days)

metrics = evaluate_and_plot_forecast(pegel_data, forecast_values)