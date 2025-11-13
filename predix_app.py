import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime

# ------------------- STREAMLIT CONFIG -------------------
st.set_page_config(page_title="PREDIX PRO", layout="wide", page_icon="🛰️")
st.markdown("<h1 style='text-align:center; color:#39FF14;'>🛰️ PREDIX PRO v4.0 – Advanced LSTM Stock Predictor</h1>", unsafe_allow_html=True)

# ------------------- SIDEBAR INPUT -------------------
st.sidebar.header("🔍 Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, INFY.NS)", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# ------------------- DATA FETCH -------------------
data = yf.download(ticker, start=start_date, end=end_date)
if data.empty:
    st.error("No data found. Please check ticker or date range.")
    st.stop()

st.success(f"✅ Loaded {ticker} data from {start_date} to {end_date}")

# ------------------- TECHNICAL INDICATORS -------------------
data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()

# RSI
delta = data["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))

# MACD
ema12 = data["Close"].ewm(span=12, adjust=False).mean()
ema26 = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"] = ema12 - ema26
data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

# ------------------- TABS -------------------
tabs = st.tabs(["📊 Overview", "📈 Technical Charts", "🤖 AI Prediction", "🔮 Forecast"])

# ------------------- TAB 1: OVERVIEW -------------------
with tabs[0]:
    st.subheader("Stock Overview Data")
    st.dataframe(data.tail())
    
    st.subheader("💹 Price Chart (with EMA)")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(data["Close"], color="#39FF14", linewidth=2, label="Close")
    ax.plot(data["EMA20"], color="#FF10F0", linestyle="--", label="EMA20")
    ax.plot(data["EMA50"], color="#00FFFF", linestyle="--", label="EMA50")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.tick_params(colors="white")
    ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
    st.pyplot(fig)

# ------------------- TAB 2: TECHNICAL CHARTS -------------------
with tabs[1]:
    st.subheader("📉 RSI (Relative Strength Index)")
    fig1, ax1 = plt.subplots(figsize=(10,3))
    ax1.plot(data["RSI"], color="#39FF14", linewidth=1.5)
    ax1.axhline(70, color="red", linestyle="--")
    ax1.axhline(30, color="blue", linestyle="--")
    ax1.set_facecolor("black")
    fig1.patch.set_facecolor("black")
    ax1.tick_params(colors="white")
    st.pyplot(fig1)

    st.subheader("📊 MACD (Moving Average Convergence Divergence)")
    fig2, ax2 = plt.subplots(figsize=(10,3))
    ax2.plot(data["MACD"], color="#FF10F0", label="MACD", linewidth=1.5)
    ax2.plot(data["Signal"], color="#00FFFF", label="Signal", linewidth=1.5)
    ax2.legend(facecolor="black", edgecolor="white", labelcolor="white")
    ax2.set_facecolor("black")
    fig2.patch.set_facecolor("black")
    ax2.tick_params(colors="white")
    st.pyplot(fig2)

st.subheader("📦 Volume Chart")

# Fix for weird multi-dimensional Volume column
if isinstance(data, (list, np.ndarray)):
    data = pd.DataFrame(data)

if "Volume" not in data.columns:
    st.warning("⚠️ Volume data not found.")
else:
    # Extract numeric and flatten
    vol = pd.Series(np.ravel(data["Volume"].values)).astype(float)

    fig3, ax3 = plt.subplots(figsize=(10,3))
    ax3.bar(range(len(vol)), vol, color="#39FF14", width=0.8)
    ax3.set_facecolor("black")
    fig3.patch.set_facecolor("black")
    ax3.tick_params(colors="white")
    ax3.set_ylabel("Volume", color="white")
    st.pyplot(fig3)


# ------------------- TAB 3: AI PREDICTION -------------------
with tabs[2]:
    st.subheader("🤖 LSTM Prediction Model")
    
    # Prepare Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    def create_dataset(dataset, look_back=60):
        X, y = [], []
        for i in range(look_back, len(dataset)):
            X.append(dataset[i - look_back:i, 0])
            y.append(dataset[i, 0])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Build Model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    with st.spinner("🚀 Training model... Please wait"):
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(real_prices, predictions))
    
    st.success(f"✅ Model trained successfully | RMSE: {rmse:.2f}")

    fig_pred, ax_pred = plt.subplots(figsize=(10,4))
    ax_pred.plot(real_prices, color="white", label="Real")
    ax_pred.plot(predictions, color="#39FF14", label="Predicted")
    ax_pred.set_facecolor("black")
    fig_pred.patch.set_facecolor("black")
    ax_pred.legend(facecolor="black", edgecolor="white", labelcolor="white")
    ax_pred.tick_params(colors="white")
    st.pyplot(fig_pred)

# ------------------- TAB 4: FORECAST -------------------
# 30-day forecast
forecast_input = scaled_data[-60:].reshape(1, 60, 1)
predictions = []

for _ in range(30):
    pred = model.predict(forecast_input, verbose=0)[0, 0]
    predictions.append(pred)
    # shift window and add new prediction
    forecast_input = np.append(forecast_input[:, 1:, :], [[[pred]]], axis=1)

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
future_dates = [data.index[-1] + datetime.timedelta(days=i + 1) for i in range(30)]
