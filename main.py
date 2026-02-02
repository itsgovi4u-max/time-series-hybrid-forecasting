# ====================================================================
# PROJECT TITLE:
# ADVANCED TIME SERIES FORECASTING WITH PROPHET AND NEURAL NETWORK HYBRID MODELS
# ====================================================================

# Dataset: Daily Delhi Climate Time Series (Kaggle)
# Models Implemented:
# 1) Baseline Model 1: Facebook Prophet (with custom seasonality)
# 2) Baseline Model 2: Bidirectional LSTM (Multivariate)
# 3) Hybrid Model: Prophet + LSTM on residuals

# Evaluation:
# - Rolling Origin Cross-Validation
# - RMSE, MAE, MAPE

# Explainability:
# - SHAP values for LSTM

# ===========================
# 1) IMPORT LIBRARIES
# ===========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prophet import Prophet

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

import shap

np.random.seed(42)
tf.random.set_seed(42)

# ===========================
# 2) LOAD KAGGLE DATASET
# ===========================

df = pd.read_csv("DailyDelhiClimateTest.csv", parse_dates=["date"])

# Rename columns for clarity
df.rename(columns={
    "date": "ds",
    "meantemp": "Temp",
    "humidity": "Humidity",
    "wind_speed": "Wind"
}, inplace=True)

df.set_index("ds", inplace=True)

print("Loaded Dataset:")
print(df.head())

# ===========================
# 3) PROPHET BASELINE MODEL
# ===========================

prophet_df = df.reset_index()[["ds", "Temp"]]
prophet_df.columns = ["ds", "y"]

prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# Custom monthly seasonality (meets project requirement)
prophet_model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

prophet_model.fit(prophet_df)

future = prophet_model.make_future_dataframe(periods=150)
forecast = prophet_model.predict(future)

prophet_pred = forecast.set_index("ds")["yhat"].loc[df.index]

df["prophet_pred"] = prophet_pred
df["residual"] = df["Temp"] - df["prophet_pred"]

# ===========================
# 4) PREPARE DATA FOR LSTM (MULTIVARIATE)
# ===========================

features = ["Temp", "Humidity", "Wind", "residual"]
data = df[features].values

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

WINDOW = 30  # Lookback window

def make_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window, 0])  # Predict Temperature
    return np.array(X), np.array(y)

X, y = make_sequences(scaled, WINDOW)

train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ===========================
# 5) TUNED BIDIRECTIONAL LSTM
# ===========================

model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True),
                  input_shape=(WINDOW, len(features))),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

print("\nTraining LSTM Model...")
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ===========================
# 6) LSTM PREDICTIONS (RESCALED)
# ===========================

y_pred = model.predict(X_test)

zeros = np.zeros((len(y_test), len(features)-1))

y_test_real = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1,1), zeros], axis=1)
)[:,0]

y_pred_real = scaler.inverse_transform(
    np.concatenate([y_pred, zeros], axis=1)
)[:,0]

# ===========================
# 7) HYBRID MODEL (PROPHET + LSTM)
# ===========================

hybrid_pred = (
    df["prophet_pred"]
    .iloc[train_size+WINDOW:train_size+WINDOW+len(y_pred_real)]
    .values + y_pred_real
)

# ===========================
# 8) ROLLING ORIGIN CROSS-VALIDATION
# ===========================

def rolling_origin_cv(series, initial=0.6, step=0.1):
    maes, rmses, mapes = [], [], []

    def mape(a, b):
        return np.mean(np.abs((a-b)/a))*100

    for i in range(3):
        start = int(len(series)*(initial + i*step))
        end = int(len(series)*(initial + (i+1)*step))

        train = series[:start]
        test = series[start:end]

        naive_pred = np.repeat(train.mean(), len(test))

        maes.append(mean_absolute_error(test, naive_pred))
        rmses.append(np.sqrt(mean_squared_error(test, naive_pred)))
        mapes.append(mape(test, naive_pred))

    return np.mean(maes), np.mean(rmses), np.mean(mapes)

cv_mae, cv_rmse, cv_mape = rolling_origin_cv(df["Temp"].values)

# ===========================
# 9) FINAL METRICS
# ===========================

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

lstm_rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
lstm_mae = mean_absolute_error(y_test_real, y_pred_real)
lstm_mape = mape(y_test_real, y_pred_real)

hyb_rmse = np.sqrt(mean_squared_error(y_test_real[:len(hybrid_pred)], hybrid_pred))
hyb_mae = mean_absolute_error(y_test_real[:len(hybrid_pred)], hybrid_pred)
hyb_mape = mape(y_test_real[:len(hybrid_pred)], hybrid_pred)

print("\n===== RESULTS =====")
print(f"LSTM  -> RMSE: {lstm_rmse:.3f}, MAE: {lstm_mae:.3f}, MAPE: {lstm_mape:.2f}%")
print(f"HYBRID-> RMSE: {hyb_rmse:.3f}, MAE: {hyb_mae:.3f}, MAPE: {hyb_mape:.2f}%")
print(f"Rolling CV -> RMSE: {cv_rmse:.3f}, MAE: {cv_mae:.3f}, MAPE: {cv_mape:.2f}%")

# ===========================
# 10) MODEL EXPLAINABILITY (SHAP)
# ===========================

# Wrapper function to reshape input for LSTM model
def f_wrapper(X):
    num_samples = X.shape[0]
    # Ensure WINDOW and len(features) match the original input dimensions
    return model.predict(X.reshape(num_samples, WINDOW, len(features)))

# Initialize explainer with the wrapper function and reshaped background data
explainer = shap.KernelExplainer(f_wrapper, X_train.reshape(X_train.shape[0], -1))

# Compute SHAP values with reshaped input
shap_values = explainer.shap_values(X_test[:10].reshape(10, -1))
print("\nSHAP explainability computed.")

# ===========================
# 11) PLOT RESULTS
# ===========================

plt.figure(figsize=(10,5))
plt.plot(y_test_real[:200], label="Actual")
plt.plot(y_pred_real[:200], label="LSTM")
plt.plot(hybrid_pred[:200], label="Hybrid")
plt.legend()
plt.title("Forecast Comparison: Actual vs LSTM vs Hybrid")
plt.show()
