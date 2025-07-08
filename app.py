import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import pandas as pd

st.set_page_config(layout="wide")
st.title("📈 توقع أسعار سهم Apple باستخدام Prophet")

# تحميل البيانات
@st.cache_data
def load_data():
    df = yf.download("AAPL", start="2018-01-01", end="2024-01-01")
    df.reset_index(inplace=True)
    df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    return df

df = load_data()
st.write("### بيانات السعر التاريخي", df.tail())

# تدريب Prophet
model = Prophet()
model.fit(df)

# توقع 30 يوم قدام
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# رسم التوقع
st.write("### التوقع للأيام القادمة")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

# رسم الاتجاه والموسمية
st.write("### تحليل الاتجاه والموسمية")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)
