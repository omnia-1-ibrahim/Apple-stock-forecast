import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import pandas as pd
import plotly.graph_objs as go

# إعداد الصفحة
st.set_page_config(page_title="📈 توقع سعر سهم Apple", layout="wide")
st.title("📈 توقع أسعار سهم Apple باستخدام Prophet")

# تحميل وتنظيف البيانات
@st.cache_data
def load_data():
    df = yf.download("AAPL", start="2018-01-01", end="2024-01-01")

    if df.empty:
        st.error("❌ فشل تحميل البيانات! تأكد من الاتصال أو الرمز.")
        st.stop()

    df.reset_index(inplace=True)

    # الاحتفاظ بالتاريخ والإغلاق فقط
    df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # تحويل التاريخ وضمان النوع الصحيح
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # حذف الصفوف اللي فيها قيم ناقصة
    df.dropna(inplace=True)

    return df

# تحميل البيانات
df = load_data()

# عرض معلومات للتأكد
st.subheader("📊 عرض أول البيانات بعد التنظيف:")
st.write(df.head())

st.success(f"✅ عدد الصفوف الجاهزة للتدريب: {df.shape[0]}")

# تدريب النموذج
try:
    model = Prophet()
    model.fit(df)

    # توقع 30 يوم قادمين
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # رسم التوقعات
    st.subheader("📉 التوقع للأيام القادمة")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    # رسم المكونات (الاتجاه والموسمية)
    st.subheader("📊 الاتجاه والموسمية")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

except Exception as e:
    st.error(f"حدث خطأ أثناء التدريب أو التوقع: {e}")
