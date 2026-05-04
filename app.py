import streamlit as st
import numpy as np
import joblib
import os

# ================== Page Config ==================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# ================== Custom CSS ==================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0d0d0d;
        color: #f0f0f0;
    }

    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #ff4b4b !important;
    }

    .header-box {
        background: linear-gradient(135deg, #1a0000, #2d0000);
        border: 1px solid #ff4b4b44;
        border-left: 4px solid #ff4b4b;
        padding: 24px 28px;
        border-radius: 8px;
        margin-bottom: 28px;
    }

    .header-box h1 {
        font-size: 1.8rem;
        margin: 0 0 6px 0;
    }

    .header-box p {
        color: #aaa;
        font-size: 0.9rem;
        margin: 0;
    }

    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #ff4b4b;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
        margin-top: 24px;
    }

    .result-box-positive {
        background: linear-gradient(135deg, #1a0000, #2d0000);
        border: 2px solid #ff4b4b;
        border-radius: 10px;
        padding: 28px;
        text-align: center;
        margin-top: 20px;
    }

    .result-box-negative {
        background: linear-gradient(135deg, #001a00, #002d00);
        border: 2px solid #00c853;
        border-radius: 10px;
        padding: 28px;
        text-align: center;
        margin-top: 20px;
    }

    .result-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .result-subtitle {
        font-size: 0.85rem;
        color: #aaa;
    }

    .prob-bar-container {
        background: #1e1e1e;
        border-radius: 6px;
        height: 10px;
        margin: 14px 0 6px 0;
        overflow: hidden;
    }

    .stSlider > div > div > div {
        background-color: #ff4b4b !important;
    }

    .stSelectbox > div > div {
        background-color: #1a1a1a !important;
        border-color: #333 !important;
        color: #f0f0f0 !important;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #1a1a1a !important;
        border-color: #333 !important;
        color: #f0f0f0 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #cc0000, #ff4b4b) !important;
        color: white !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 14px 0 !important;
        width: 100% !important;
        font-size: 1rem !important;
        letter-spacing: 1px !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px #ff4b4b44 !important;
    }

    .info-card {
        background: #141414;
        border: 1px solid #222;
        border-radius: 8px;
        padding: 16px;
        font-size: 0.82rem;
        color: #888;
        margin-top: 20px;
        font-family: 'IBM Plex Mono', monospace;
    }

    .divider {
        border: none;
        border-top: 1px solid #222;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ================== Load Model ==================
@st.cache_resource
def load_model():
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except:
    model_loaded = False

# ================== Header ==================
st.markdown("""
<div class="header-box">
    <h1>❤️ Heart Disease Predictor</h1>
    <p>Enter patient medical data to assess heart disease risk using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model files not found! Make sure `heart_model.pkl` and `scaler.pkl` are in the same folder.")
    st.stop()

# ================== Input Form ==================
st.markdown('<div class="section-label">// Patient Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=50, step=1)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120, step=1)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240, step=1)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    ca = st.selectbox("Major Vessels (ca)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", options=[0, 1, 2, 3], format_func=lambda x: {0: "0 - Normal", 1: "1 - Fixed Defect", 2: "2 - Reversible Defect", 3: "3 - Other"}[x])

with col2:
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3],
                      format_func=lambda x: {0: "0 - Typical Angina", 1: "1 - Atypical Angina",
                                              2: "2 - Non-Anginal Pain", 3: "3 - Asymptomatic"}[x])
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                       format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                           format_func=lambda x: {0: "0 - Normal", 1: "1 - ST-T Abnormality", 2: "2 - LV Hypertrophy"}[x])
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                         format_func=lambda x: "No" if x == 0 else "Yes")
    slope = st.selectbox("Slope of ST Segment", options=[0, 1, 2],
                         format_func=lambda x: {0: "0 - Upsloping", 1: "1 - Flat", 2: "2 - Downsloping"}[x])

# ================== Predict ==================
st.markdown('<hr class="divider">', unsafe_allow_html=True)

if st.button("🔍 ANALYZE PATIENT DATA"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    prob_positive = probability[1] * 100
    prob_negative = probability[0] * 100

    if prediction == 1:
        st.markdown(f"""
        <div class="result-box-positive">
            <div class="result-title" style="color:#ff4b4b">⚠️ HIGH RISK DETECTED</div>
            <div class="result-subtitle">Model predicts presence of heart disease</div>
            <div style="margin-top:16px; font-family:'IBM Plex Mono',monospace; font-size:2rem; color:#ff4b4b; font-weight:600">
                {prob_positive:.1f}%
            </div>
            <div style="font-size:0.8rem; color:#888">probability of heart disease</div>
            <div class="prob-bar-container">
                <div style="background:#ff4b4b; height:100%; width:{prob_positive}%; border-radius:6px;"></div>
            </div>
            <div style="font-size:0.78rem; color:#666; margin-top:8px">
                ⚕️ Please consult a cardiologist immediately
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box-negative">
            <div class="result-title" style="color:#00c853">✅ LOW RISK</div>
            <div class="result-subtitle">Model predicts no significant heart disease risk</div>
            <div style="margin-top:16px; font-family:'IBM Plex Mono',monospace; font-size:2rem; color:#00c853; font-weight:600">
                {prob_negative:.1f}%
            </div>
            <div style="font-size:0.8rem; color:#888">probability of no heart disease</div>
            <div class="prob-bar-container">
                <div style="background:#00c853; height:100%; width:{prob_negative}%; border-radius:6px;"></div>
            </div>
            <div style="font-size:0.78rem; color:#666; margin-top:8px">
                ✅ Continue regular health checkups
            </div>
        </div>
        """, unsafe_allow_html=True)

# ================== Footer ==================
st.markdown("""
<div class="info-card">
    ⚠️  DISCLAIMER: This tool is for educational purposes only.<br>
    It is NOT a substitute for professional medical advice.<br><br>
    // Model: Random Forest Classifier | AUC: 0.88 | Dataset: UCI Heart Disease
</div>
""", unsafe_allow_html=True)