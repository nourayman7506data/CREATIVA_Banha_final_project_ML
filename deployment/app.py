# ============================================================
# app.py
# ------------------------------------------------------------
# واجهة Streamlit للتنبؤ باشتراك العميل في Term Deposit
# (تم حذف خيار unknown من جميع القوائم بناءً على طلبك)
# ============================================================

import streamlit as st
import pandas as pd
import joblib

# استيراد الكلاس المخصص لفك تشفير pickle
from pipeline_utils import IQRCapper  # noqa: F401

# ------------------------------------------------------------
# إعدادات الصفحة
# ------------------------------------------------------------
st.set_page_config(
    page_title="Bank Deposit Prediction",
    page_icon="🏦",
    layout="centered",
)

# ------------------------------------------------------------
# CSS لتعديل الألوان وضمان وضوح النصوص في الـ Dark Mode
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');

    html, body, [class*="css"], label, input, button {
        font-family: 'Tajawal', sans-serif !important;
        direction: rtl;
    }

    .header-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        padding: 25px 20px;
        border-radius: 16px;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    }
    .header-title {
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 8px;
        color: #FFFFFF !important;
    }
    .header-subtitle {
        font-size: 14px;
        opacity: 0.9;
        line-height: 1.6;
        color: #E2E8F0 !important;
    }

    .stSelectbox label, 
    .stNumberInput label, 
    div[data-testid="stMarkdownContainer"] p,
    label p {
        font-size: 15px !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
        margin-bottom: 6px !important;
    }

    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div {
        border-radius: 8px !important;
        border: 1px solid #475569 !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 0 !important;
        margin-top: 10px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.6);
        color: #FFFFFF !important;
    }
    </style>

    <div class="header-box">
        <div class="header-title">🏦 توقع اشتراك العميل في الوديعة (Term Deposit)</div>
        <div class="header-subtitle">
            قم بإدخال بيانات العميل وتفاصيل آخر تواصل للتحليل والتنبؤ باستخدام موديل <b>LightGBM</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# تحميل الموديل
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model_pipeline.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ ملف الموديل model_pipeline.pkl غير موجود!")
    st.stop()


# ------------------------------------------------------------
# نموذج إدخال البيانات (القيم الفعلية فقط بدون unknown)
# ------------------------------------------------------------

# 1. البيانات الشخصية والمالية
with st.container(border=True):
    st.subheader("👤 البيانات الشخصية والمالية")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("العمر (Age)", min_value=18, max_value=95, value=40)
        
        # 🎯 تم حذف unknown من الوظيفة
        job = st.selectbox(
            "الوظيفة (Job)",
            ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
             "retired", "self-employed", "services", "student", "technician",
             "unemployed"],
        )
        
        marital = st.selectbox("الحالة الاجتماعية (Marital)", ["married", "single", "divorced"])
        
        # 🎯 تم حذف unknown من المستوى التعليمي
        education = st.selectbox("المستوى التعليمي (Education)", ["primary", "secondary", "tertiary"])

    with col2:
        default = st.selectbox("تعثر سداد سابق؟ (Default)", ["no", "yes"])
        balance = st.number_input("متوسط الرصيد السنوي باليورو (Balance)", value=500, step=100)
        housing = st.selectbox("قرض عقاري؟ (Housing)", ["no", "yes"])
        loan = st.selectbox("قرض شخصي؟ (Loan)", ["no", "yes"])

# 2. تفاصيل الاتصال والحملة
with st.container(border=True):
    st.subheader("📞 تفاصيل التواصل والحملة")
    col3, col4 = st.columns(2)

    with col3:
        # 🎯 تم حذف unknown من وسيلة الاتصال
        contact = st.selectbox("وسيلة الاتصال (Contact)", ["cellular", "telephone"])
        
        day = st.number_input("يوم الاتصال بالشهر (Day)", min_value=1, max_value=31, value=15)
        
        month = st.selectbox(
            "شهر آخر اتصال (Month)",
            ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        )
        
        duration = st.number_input("مدة آخر مكالمة بالثواني (Duration)", min_value=0, value=180)

    with col4:
        campaign = st.number_input("عدد مرات التواصل بالحملة (Campaign)", min_value=1, value=2)
        pdays = st.number_input("الأيام منذ آخر تواصل سابق (Pdays)", value=-1)
        previous = st.number_input("عدد مرات التواصل السابق (Previous)", min_value=0, value=0)
        
        # 🎯 تم حذف unknown من نتيجة الحملة السابقة
        poutcome = st.selectbox("نتيجة الحملة السابقة (Poutcome)", ["failure", "other", "success"])


# ------------------------------------------------------------
# زر التوقع وعرض النتيجة
# ------------------------------------------------------------
if st.button("🔮 بدء التوقع الآن", use_container_width=True):

    input_df = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.write("")
    with st.container(border=True):
        st.subheader("📊 النتيجة:")
        
        c1, c2 = st.columns([3, 1])
        with c1:
            if prediction == 1:
                st.success("✅ **من المتوقع أن العميل سيشترك في الوديعة (Term Deposit)**")
            else:
                st.error("❌ **من المتوقع أن العميل لن يشترك في الوديعة (Term Deposit)**")
        
        with c2:
            st.metric("احتمال الاشتراك", f"{probability * 100:.1f}%")

        st.progress(float(probability))