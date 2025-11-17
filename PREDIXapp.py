# Advanced Stock Prediction with LSTM Neural Network
# Complete Package with MACD, RSI & Neon Charts + 30-Day Min/Max Forecast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ---------------- Plotting Style ----------------
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

# ---------------- Technical Indicators ----------------
def calculate_technical_indicators(data):
    df = data.copy()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 0.0001)
    df['RSI'] = 100 - (100 / (1 + rs))
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

# ---------------- Recommended Stocks ----------------
def plot_recommended_companies():
    companies = [
        ["Palantir Technologies", "PLTR", "Technology", 109.4, "Strong Buy"],
        ["GE Vernova", "GEV", "Energy", 100.7, "Strong Buy"],
        ["Super Micro Computer", "SMCI", "Technology", 93.5, "Strong Buy"],
        ["NRG Energy", "NRG", "Energy", 85.3, "Strong Buy"],
        ["Seagate Technology", "STX", "Technology", 81.9, "Strong Buy"],
        ["Western Digital", "WDC", "Technology", 74.6, "Strong Buy"],
        ["Newmont Corp.", "NEM", "Materials", 66.8, "Strong Buy"],
        ["Tapestry", "TPR", "Consumer Discretionary", 65.4, "Strong Buy"],
        ["Howmet Aerospace", "HWM", "Industrials", 64.4, "Strong Buy"],
        ["General Electric", "GE", "Industrials", 62.5, "Strong Buy"]
    ]
    fig, ax = plt.subplots(figsize=(14,6))
    ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=companies,
                     colLabels=["Company", "Ticker", "Sector", "Current Price (USD)", "Analyst Rating"],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    for (i,j), cell in table.get_celld().items():
        if i==0: 
            cell.set_facecolor("#0D1B2A"); cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor("#1B263B" if i%2==1 else "#415A77"); cell.set_text_props(color='white')
    plt.title("📈 Recommended Stocks for Potential Growth", fontsize=16, color='white', pad=20)
    fig.patch.set_facecolor('#0D1B2A')
    plt.show()

# ---------------- LSTM Model ----------------
def create_lstm_model():
    print("🧠 Advanced Stock Prediction with LSTM Neural Network")
    print("=" * 60)
    
    symbol = input("Enter stock symbol (e.g., AAPL, MSFT, TATAMOTORS.NS): ").upper().strip()
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    
    stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if len(stock_data) < 100:
        print("Error: Not enough data."); return
    stock_data = calculate_technical_indicators(stock_data)
    
    # Scale data
    data = stock_data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    
    # Create lookback sequences
    lookback = 60
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i,0])
        y.append(scaled_data[i,0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Train-test split
    split = int(0.8*len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback,1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Training LSTM model...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    
    # ---------------- Model Predictions ----------------
    train_predict = scaler.inverse_transform(model.predict(X_train))
    test_predict = scaler.inverse_transform(model.predict(X_test))
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1,1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))
    
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
    
    # 30-day forecast
    last_60_days = scaled_data[-lookback:]
    predictions = []
    for i in range(30):
        x_input = last_60_days.reshape(1, lookback,1)
        pred = model.predict(x_input, verbose=0)
        predictions.append(pred[0,0])
        last_60_days = np.append(last_60_days[1:], pred)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
    future_dates = [stock_data.index[-1] + timedelta(days=i+1) for i in range(30)]
    
    # ---------------- Plots ----------------
    # 1. Training Progress
    plt.figure(figsize=(16,5))
    plt.plot(history.history['loss'], label='Training Loss', color='#39FF14', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='#0FF0FC', linewidth=2)
    plt.title(f'{symbol} LSTM Model Training Progress', fontsize=16)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(alpha=0.3); plt.show()
    
    # 2. Test Set Predictions vs Actual
    plt.figure(figsize=(16,5))
    test_index = range(len(data)-len(test_predict), len(data))
    plt.plot(stock_data.index[test_index], y_test_actual, label='Actual Price', color='#FF073A', linewidth=2)
    plt.plot(stock_data.index[test_index], test_predict, label='Predicted Price', color='#39FF14', linewidth=2)
    plt.title(f'{symbol} Test Set Predictions vs Actual', fontsize=16)
    plt.xlabel('Date'); plt.ylabel('Price ($)')
    plt.legend(); plt.grid(alpha=0.3); plt.show()
    
    # 3. Historical + Forecast
    plt.figure(figsize=(16,6))
    plt.plot(stock_data.index, data, label='Historical', color='#FF073A', linewidth=2)
    plt.plot(stock_data.index[lookback:lookback+len(train_predict)], train_predict, label='Train Predict', color='#39FF14', linewidth=2)
    plt.plot(stock_data.index[test_index], test_predict, label='Test Predict', color='#0FF0FC', linewidth=2)
    plt.plot(future_dates, predictions, label='30-Day Forecast', color='#F3F315', linewidth=3)
    plt.title(f'{symbol} Stock Prediction with LSTM', fontsize=16)
    plt.xlabel('Date'); plt.ylabel('Price ($)')
    plt.legend(); plt.grid(alpha=0.3); plt.show()
    
    # 4. MACD
    plt.figure(figsize=(16,4))
    plt.plot(stock_data.index, stock_data['MACD'], label='MACD', color='#FF073A', linewidth=2)
    plt.plot(stock_data.index, stock_data['MACD_Signal'], label='Signal Line', color='#39FF14', linewidth=2)
    plt.bar(stock_data.index, stock_data['MACD_Hist'], label='Histogram', color='#0FF0FC', alpha=0.3)
    plt.title(f'{symbol} MACD Indicator', fontsize=14)
    plt.xlabel('Date'); plt.ylabel('Value')
    plt.legend(); plt.grid(alpha=0.3); plt.show()
    
    # 5. RSI
    plt.figure(figsize=(16,4))
    plt.plot(stock_data.index, stock_data['RSI'], label='RSI', color='#F3F315', linewidth=2)
    plt.axhline(70, color='red', linestyle='--', alpha=0.7)
    plt.axhline(30, color='green', linestyle='--', alpha=0.7)
    plt.title(f'{symbol} RSI Indicator', fontsize=14)
    plt.xlabel('Date'); plt.ylabel('Value')
    plt.legend(); plt.grid(alpha=0.3); plt.show()
    
    # ---------------- Metrics & Forecast ----------------
    print("\n📊 Model Metrics:")
    print(f"Training RMSE: ${train_rmse:.2f}")
    print(f"Testing RMSE: ${test_rmse:.2f}")
    
    # 30-Day Forecast Min & Max
    min_price = predictions.min()
    max_price = predictions.max()
    print(f"\n🔮 30-Day Forecast Summary for {symbol}:")
    print(f"Current Price: ${data[-1][0]:.2f}")
    print(f"Predicted Price in 30 days: ${predictions[-1][0]:.2f}")
    print(f"Minimum Predicted Price: ${min_price:.2f}")
    print(f"Maximum Predicted Price: ${max_price:.2f}")
    
    # Next 7 days
    print("\nNext 7 Days Forecast:")
    for i in range(7):
        print(f"{future_dates[i].strftime('%Y-%m-%d')}: ${predictions[i][0]:.2f}")
    
    # Save model
    model.save('my_stock_prediction_model.h5')
    print("\n💾 Model saved as 'my_stock_prediction_model.h5'")
    
    # Recommended companies
    plot_recommended_companies()
    
    # Disclaimer
    print("\n⚠️  DISCLAIMER:")
    print("   This prediction is based on historical patterns only.")
    print("   It does not consider future news or market events.")
    print("   Use for educational purposes only.")

# Run the LSTM model
create_lstm_model()



