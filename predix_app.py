# predix_app.py
# ---------------------------------------------------
# PREDIX - AI Stock Predictor (LSTM-based)
# Futuristic neon-dark themed Streamlit app
# ---------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from geopy.geocoders import Nominatim
import pydeck as pdk
import time

# -----------------------------------
# APP CONFIG
# -----------------------------------
st.set_page_config(page_title="PREDIX - AI Stock Predictor", page_icon="💹", layout="wide")

# Custom CSS for neon-dark theme
st.markdown("""
    <style>
    body {
        background-color: #0f1116;
        color: #e8e8e8;
    }
    [data-testid="stSidebar"] {
        background-color: #12151c;
    }
    .main-title {
        font-size: 2.8em;
        text-align: center;
        color: #00FFFF;
        text-shadow: 0 0 25px #00FFFF;
        margin-bottom: 10px;
    }
    .sub-text {
        text-align: center;
        font-size: 1.2em;
        color: #aaa;
        margin-bottom: 25px;
    }
    .card {
        background-color: #1b1f27;
        border-radius: 15px;
        padding: 25px;
        margin-top: 25px;
        box-shadow: 0 0 20px rgba(0,255,255,0.1);
    }
    .section-title {
        color: #7df9ff;
        font-size: 1.5em;
        margin-bottom: 10px;
        text-shadow: 0 0 10px #00FFFF;
    }
    .auto-save {
        text-align: right;
        font-size: 0.9em;
        color: #8f8;
        animation: blink 2s infinite;
    }
    @keyframes blink {
        50% {opacity: 0.4;}
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------
# TITLE
# -----------------------------------
st.markdown("<div class='main-title'>PREDIX 💹</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Neural-Powered Stock Predictor with LSTM Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='auto-save'>💾 Auto-saving progress...</div>", unsafe_allow_html=True)

# -----------------------------------
# SIDEBAR INPUT
# -----------------------------------
st.sidebar.header("🔍 Stock Prediction Settings")
company_name = st.sidebar.text_input("Enter Company Name", "Apple")
start_date = st.sidebar.date_input("Start Date", dt.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date.today())

# Map company name to ticker
ticker_map = {
    "Apple": "AAPL",
    "Google": "GOOG",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Meta": "META",
    "Reliance": "RELIANCE.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS"
}
ticker = ticker_map.get(company_name, None)

if not ticker:
    st.error("❌ Company not found in the database. Try a different name.")
    st.stop()

# -----------------------------------
# FETCH STOCK DATA
# -----------------------------------
data_load_state = st.text("Loading stock data...")
data = yf.download(ticker, start=start_date, end=end_date)
data_load_state.text("✅ Stock data loaded successfully!")

# -----------------------------------
# VISUALIZATION 1: STOCK PRICE HISTORY
# -----------------------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Stock Price History</div>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close', line=dict(color='#00FFFF')))
    fig.update_layout(template='plotly_dark', title=f"{company_name} Stock Price History", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("🧠 This chart shows the company's historical closing prices, providing context for prediction.", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------
# DATA PREPARATION
# -----------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]

x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i])
    y_train.append(train_data[i])
x_train, y_train = np.array(x_train), np.array(y_train)

# -----------------------------------
# LSTM MODEL
# -----------------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
with st.spinner('Training the LSTM model...'):
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

# -----------------------------------
# PREDICTION
# -----------------------------------
x_test, y_test = [], scaled_data[train_size:]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i])
x_test = np.array(x_test)
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# -----------------------------------
# VISUALIZATION 2: PREDICTIONS
# -----------------------------------
train = data[:train_size]
valid = data[train_size:]
valid['Predictions'] = predictions

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🚀 Predicted vs Actual Prices</div>", unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Train', line=dict(color='gray')))
    fig2.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name='Actual', line=dict(color='#00FFFF')))
    fig2.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name='Predicted', line=dict(color='#FF00FF')))
    fig2.update_layout(template='plotly_dark', title=f"{company_name} - LSTM Predictions", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("💡 The magenta line shows PREDIX’s LSTM predictions versus the actual stock prices.", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------
# VISUALIZATION 3: FORECAST
# -----------------------------------
future_days = 30
last_60_days = scaled_data[-60:]
forecast_input = last_60_days.reshape(1, -1, 1)

predicted_future = []
for _ in range(future_days):
    pred = model.predict(forecast_input)
    predicted_future.append(pred[0, 0])
    forecast_input = np.append(forecast_input[:, 1:, :], [[[pred]]], axis=1)

predicted_future = scaler.inverse_transform(np.array(predicted_future).reshape(-1, 1))

future_dates = pd.date_range(end_date, periods=future_days+1)[1:]
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_future.flatten()})

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔮 30-Day Future Forecast</div>", unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted Price'],
                              mode='lines+markers', line=dict(color='#00FFFF')))
    fig3.update_layout(template='plotly_dark', title=f"{company_name} - Next 30 Days Forecast", xaxis_title='Date', yaxis_title='Predicted Price')
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("📈 This forecast visualizes the projected trend for the next 30 days using the LSTM model.", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------
# COMPANY LOCATION MAP
# -----------------------------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📍 Company Headquarters</div>", unsafe_allow_html=True)
    geolocator = Nominatim(user_agent="predix")
    location = geolocator.geocode(f"{company_name} headquarters")
    if location:
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/dark-v11',
            initial_view_state=pdk.ViewState(latitude=location.latitude, longitude=location.longitude, zoom=10),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=pd.DataFrame({'lat': [location.latitude], 'lon': [location.longitude]}),
                    get_position='[lon, lat]',
                    get_color='[0, 255, 255, 160]',
                    get_radius=1000,
                )
            ],
        ))
        st.markdown(f"🏢 Headquarters located in **{location.address}**.", unsafe_allow_html=True)
    else:
        st.warning("Could not locate headquarters for this company.")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("""
    <hr style='border: 1px solid #222;'/>
    <div style='text-align:center; color:#555;'>
        © 2025 <b>PREDIX AI</b> — Built with ❤️ using LSTM & Streamlit
    </div>
""", unsafe_allow_html=True)
