import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objects as go
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

@st.cache_data
def load_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6*365)
    df = yf.download("AAPL", start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df[['Date', 'Close']]

st.set_page_config(
    page_title="📈 Apple Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Apple Stock Price Prediction")
st.write("هذا التطبيق يستخدم SARIMA و LSTM للتنبؤ بسعر سهم Apple")

df = load_data()

col1, col2 = st.columns(2)
with col1:
    model_choice = st.selectbox("اختر الموديل", ["SARIMA", "LSTM"])
with col2:
    n_days = st.slider("عدد الأيام المستقبلية للتنبؤ", 30, 180, 60)

if model_choice == "SARIMA":
    st.subheader("تنبؤ SARIMA")
    df_sarima = df.copy()
    df_sarima.set_index('Date', inplace=True)
    train_size = int(len(df_sarima) * 0.8)
    train = df_sarima.iloc[:train_size]
    test = df_sarima.iloc[train_size:]
    with st.spinner('جاري تدريب نموذج SARIMA...'):
        model = SARIMAX(train['Close'], order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        model_fit = model.fit(disp=False)
    forecast_steps = len(test) + n_days
    forecast_result = model_fit.get_forecast(steps=forecast_steps)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    test_forecast = forecast[:len(test)]
    test_actual = test['Close']
    rmse = np.sqrt(mean_squared_error(test_actual, test_forecast))
    mape = mean_absolute_percentage_error(test_actual, test_forecast) * 100
    st.write(f"📊 **RMSE**: {rmse:.2f}")
    st.write(f"📊 **MAPE**: {mape:.2f}%")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='تدريب', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test.index, y=test['Close'], mode='lines', name='اختبار', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='تنبؤ', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=forecast.index, y=conf_int.iloc[:, 0], fill=None, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast.index, y=conf_int.iloc[:, 1], fill='tonexty', mode='lines', name='نطاق الثقة 95%', line=dict(width=0), fillcolor='rgba(255, 0, 0, 0.2)'))
    last_train_date = test.index[-1]
    fig.add_vline(x=last_train_date, line_dash="dash", line_color="orange", annotation_text="بداية التنبؤ", annotation_position="top left")
    fig.update_layout(title="تنبؤ أسعار سهم Apple باستخدام SARIMA", xaxis_title="التاريخ", yaxis_title="السعر ($)", legend_title="المفتاح", hovermode="x unified", height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("التنبؤات المستقبلية")
    forecast_df = pd.DataFrame({
        'التاريخ': forecast.index[-n_days:],
        'السعر المتوقع': forecast.values[-n_days:],
        'الحد الأدنى': conf_int.iloc[-n_days:, 0].values,
        'الحد الأعلى': conf_int.iloc[-n_days:, 1].values
    })
    st.dataframe(forecast_df.style.format({'السعر المتوقع': '{:.2f}', 'الحد الأدنى': '{:.2f}', 'الحد الأعلى': '{:.2f}'}))

elif model_choice == "LSTM":
    st.subheader("تنبؤ LSTM")
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
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    with st.spinner('جاري تدريب نموذج LSTM...'):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    predicted = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    st.write(f"📊 **RMSE**: {rmse:.2f}")
    st.write(f"📊 **MAPE**: {mape:.2f}%")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'].iloc[train_size+sequence_length:train_size+sequence_length+len(actual)], y=actual.flatten(), mode='lines', name='سعر فعلي', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'].iloc[train_size+sequence_length:train_size+sequence_length+len(actual)], y=predicted.flatten(), mode='lines', name='تنبؤ LSTM', line=dict(color='red', dash='dash')))
    fig.update_layout(title="تنبؤ أسعار سهم Apple باستخدام LSTM", xaxis_title="التاريخ", yaxis_title="السعر ($)", legend_title="المفتاح", hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='فقدان التدريب'))
    fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='فقدان التحقق'))
    fig_loss.update_layout(title='فقدان النموذج أثناء التدريب', xaxis_title='العصر', yaxis_title='فقدان', height=400)
    st.plotly_chart(fig_loss, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>تم التطوير باستخدام Python, SARIMA, LSTM, Streamlit و Yahoo Finance</p>
    <p>© 2024 أداة توقع أسعار الأسهم | إعداد: Omnia Ebrahim</p>
</div>
""", unsafe_allow_html=True)
