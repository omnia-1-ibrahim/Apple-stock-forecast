import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Prophet
from prophet import Prophet

# LSTM
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------
# تحميل بيانات Apple
@st.cache_data
def load_data():
    df = yf.download("AAPL", start="2018-01-01", end="2024-01-01")
    df.reset_index(inplace=True)
    return df[['Date', 'Close']]

df = load_data()
st.title("📈 Apple Stock Price Prediction")
st.write("هذا التطبيق بيستخدم Prophet و LSTM للتنبؤ بسعر سهم Apple")

# اختيار الموديل
model_choice = st.selectbox("اختار الموديل", ["Prophet", "LSTM"])
n_days = st.slider("عدد الأيام المستقبلية للتنبؤ", 30, 180, 60)

# ----------------------------------------
# Prophet Model
if model_choice == "Prophet":
    df_prophet = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=n_days)
    forecast = model.predict(future)
    
    # دمج البيانات
    df_forecast = forecast[['ds', 'yhat']].merge(df_prophet, on='ds', how='left')
    df_forecast.dropna(inplace=True)
    rmse = np.sqrt(mean_squared_error(df_forecast['y'], df_forecast['yhat']))

    # رسم
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_forecast['ds'], df_forecast['y'], label='Actual Price', color='black')
    ax.plot(df_forecast['ds'], df_forecast['yhat'], label='Prophet Predicted Price', color='blue')
    ax.set_title("Prophet Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
    st.write(f"📊 RMSE = {rmse:.2f}")

# ----------------------------------------
# LSTM Model
elif model_choice == "LSTM":
    data = df[['Close']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # بناء الموديل
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # توقع
    predicted = model.predict(X)
    predicted = scaler.inverse_transform(predicted)
    actual = scaler.inverse_transform(y.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # رسم
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(actual, color='black', label='Actual Price')
    ax2.plot(predicted, color='green', label='LSTM Predicted Price')
    ax2.set_title("LSTM Prediction")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Price")
    ax2.legend()
    st.pyplot(fig2)
    st.write(f"📊 RMSE = {rmse:.2f}")
