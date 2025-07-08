import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="📈 توقع أسعار الأسهم",
    page_icon="📈",
    layout="wide"
)

# عنوان التطبيق
st.title("📈 توقع أسعار الأسهم باستخدام Prophet")
st.markdown("""
<div style="text-align: right; direction: rtl;">
    <p>هذا التطبيق يتنبأ بأسعار الأسهم باستخدام خوارزمية <strong>Facebook Prophet</strong></p>
    <p>لبدء الاستخدام، اختر السهم والفترة الزمنية ثم انقر على زر "تنبؤ"</p>
</div>
""", unsafe_allow_html=True)

# شريط جانبي للإعدادات
with st.sidebar:
    st.header("⚙️ إعدادات النموذج")
    
    # اختيار السهم
    ticker_options = {
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Tesla': 'TSLA',
        'Google': 'GOOGL',
        'Amazon': 'AMZN',
        'Meta (Facebook)': 'META'
    }
    selected_ticker = st.selectbox(
        "اختر السهم",
        list(ticker_options.keys()),
        index=0
    )
    ticker = ticker_options[selected_ticker]
    
    # اختيار فترة البيانات
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "تاريخ البداية",
            value=datetime.now() - timedelta(days=365*5),
            max_value=datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input(
            "تاريخ النهاية",
            value=datetime.now()
        )
    
    # إعدادات النموذج
    st.subheader("معاملات النموذج")
    forecast_period = st.slider("عدد أيام التنبؤ", 30, 365, 90)
    confidence_interval = st.slider("فترة الثقة", 0.80, 0.99, 0.95, step=0.01)
    
    # زر التنبؤ
    run_forecast = st.button("🚀 تنبؤ", use_container_width=True)

# قسم النتائج الرئيسي
if run_forecast:
    with st.spinner("جاري جلب البيانات وتدريب النموذج..."):
        try:
            # جلب البيانات من Yahoo Finance
            df = yf.download(ticker, start=start_date, end=end_date)
            
            # التحقق من وجود بيانات كافية
            if df.empty:
                st.error("⚠️ لا توجد بيانات متاحة لهذه الفترة. الرجاء اختيار تواريخ مختلفة.")
                st.stop()
                
            if len(df) < 10:
                st.error(f"⚠️ البيانات قليلة جداً ({len(df)} صفوف فقط). الرجاء اختيار فترة زمنية أطول.")
                st.stop()
                
            # إعادة تسمية الأعمدة لتناسب Prophet
            df = df.reset_index()
            df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
            df = df[['ds', 'y']].dropna()
            
            # تقسيم البيانات للتدريب والاختبار
            train_size = int(len(df) * 0.8)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            
            # تدريب النموذج
            model = Prophet(
                daily_seasonality=False,
                yearly_seasonality=True,
                weekly_seasonality=True,
                interval_width=confidence_interval
            )
            model.fit(train_df)
            
            # إنشاء إطار للتنبؤ
            future = model.make_future_dataframe(periods=forecast_period + len(test_df))
            forecast = model.predict(future)
            
            # حساب دقة النموذج
            test_forecast = forecast.iloc[train_size:train_size + len(test_df)]
            test_forecast = test_forecast.set_index('ds')
            test_df = test_df.set_index('ds')
            
            if not test_forecast.empty and not test_df.empty:
                merged = test_df.join(test_forecast[['yhat']], how='inner')
                if not merged.empty:
                    mae = mean_absolute_error(merged['y'], merged['yhat'])
                    mape = mean_absolute_percentage_error(merged['y'], merged['yhat']) * 100
                    last_price = df['y'].iloc[-1]
                    last_forecast = forecast['yhat'].iloc[-1]
                    change_percent = ((last_forecast - last_price) / last_price) * 100
                    
            # عرض النتائج
            st.success("✅ تم التدريب والتنبؤ بنجاح!")
            
            # مقاييس الأداء
            st.subheader("📊 أداء النموذج")
            if not merged.empty:
                col1, col2, col3 = st.columns(3)
                col1.metric("متوسط الخطأ المطلق (MAE)", f"${mae:.2f}")
                col2.metric("متوسط الخطأ النسبي (MAPE)", f"{mape:.2f}%")
                col3.metric("التغير المتوقع", f"{change_percent:.2f}%", f"${last_forecast:.2f}")
            else:
                st.warning("لا توجد بيانات كافية لحساب دقة النموذج")
            
            # عرض الرسوم البيانية
            st.subheader("📈 التوقعات مقابل البيانات الفعلية")
            
            # رسم بياني بالرسوم المتحركة
            fig = go.Figure()
            
            # البيانات الفعلية
            fig.add_trace(go.Scatter(
                x=df['ds'],
                y=df['y'],
                mode='lines',
                name='الأسعار الفعلية',
                line=dict(color='#1f77b4', width=3)
            ))
            
            # التوقعات
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='التنبؤ',
                line=dict(color='#ff7f0e', width=3, dash='dash')
            ))
            
            # مناطق الثقة
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                fill='tonexty',
                mode='lines',
                name=f'نطاق الثقة {int(confidence_interval*100)}%',
                line=dict(width=0),
                fillcolor='rgba(255, 127, 14, 0.2)'
            ))
            
            # خط فاصل بين البيانات التاريخية والتنبؤ
            last_train_date = train_df['ds'].max()
            fig.add_vline(
                x=last_train_date,
                line_dash="dash",
                line_color="green",
                annotation_text="بداية التنبؤ",
                annotation_position="top left"
            )
            
            # تخصيص التخطيط
            fig.update_layout(
                title=f"توقع أسعار سهم {selected_ticker} ({ticker})",
                xaxis_title="التاريخ",
                yaxis_title="سعر الإغلاق ($)",
                legend_title="المفتاح",
                hovermode="x unified",
                template="plotly_white",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # عرض جدول التنبؤات
            st.subheader("📋 جدول التنبؤات")
            forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period)
            forecast_table = forecast_table.rename(columns={
                'ds': 'التاريخ',
                'yhat': 'السعر المتوقع',
                'yhat_lower': 'الحد الأدنى',
                'yhat_upper': 'الحد الأعلى'
            })
            forecast_table['التاريخ'] = forecast_table['التاريخ'].dt.strftime('%Y-%m-%d')
            st.dataframe(forecast_table.style.format({
                'السعر المتوقع': '{:.2f}',
                'الحد الأدنى': '{:.2f}',
                'الحد الأعلى': '{:.2f}'
            }), use_container_width=True)
            
            # تحميل البيانات
            st.subheader("💾 تحميل النتائج")
            csv = forecast_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 تحميل التنبؤات كملف CSV",
                data=csv,
                file_name=f"{ticker}_forecast_{end_date.strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"حدث خطأ: {str(e)}")
            st.error("الرجاء المحاولة مرة أخرى أو تغيير الإعدادات")
else:
    # شاشة الترحيب
    st.subheader("مرحباً بك في أداة توقع أسعار الأسهم")
    st.info("""
    <div style="text-align: right; direction: rtl;">
    <p>لبدء التنبؤ:</p>
    <ol>
        <li>اختر السهم من القائمة في الشريط الجانبي</li>
        <li>حدد فترة البيانات التاريخية</li>
        <li>اضبط معاملات النموذج (اختياري)</li>
        <li>انقر على زر "تنبؤ"</li>
    </ol>
    <p>المميزات المتاحة:</p>
    <ul>
        <li>تنبؤ بأسعار الأسهم لمدة تصل إلى سنة</li>
        <li>مقارنة التوقعات مع البيانات الفعلية</li>
        <li>حساب دقة النموذج</li>
        <li>تحميل النتائج كملف CSV</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # مثال توضيحي
    st.subheader("مثال توضيحي")
    st.image("https://i.imgur.com/5Z3zQ1l.png", caption="نتيجة نموذجية للتنبؤ بأسعار الأسهم")

# تذييل الصفحة
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>تم التطوير باستخدام Python, Prophet, Streamlit و Yahoo Finance</p>
    <p>© 2024 أداة توقع أسعار الأسهم | إعداد: Omnia Ebrahim</p>
</div>
""", unsafe_allow_html=True)
