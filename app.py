import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Calories Burnt Predictor", page_icon="", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #0a0a0f 0%, #111120 40%, #0d0d1a 100%);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    * { font-family: 'Inter', sans-serif !important; }
    h1, h2, h3, h4, p, label, span, .stMarkdown, div {
        color: #e8e8f0 !important;
    }

    /* ── Hero title ── */
    .hero-title {
        text-align: left;
        padding: 20px 0 5px;
    }
    .hero-title h1 {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ff6b6b, #ff8e53, #ffd93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0 !important;
    }
    .hero-subtitle {
        text-align: left;
        font-size: 1rem;
        color: #8888aa !important;
        margin-bottom: 30px;
    }

    /* ── Glass card ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 20px;
        padding: 28px 24px;
        margin-bottom: 20px;
    }
    .card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
    }
    .card-header .icon {
        font-size: 1.3rem;
    }
    .card-header .label {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #9999bb !important;
    }


    /* ── Inputs ── */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #e0e0f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        border-radius: 12px !important;
    }
    .stNumberInput input:focus {
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15) !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 12px !important;
    }
    label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #b0b0cc !important;
    }
    
    div[data-testid="InputInstructions"] {
        position: absolute !important;
        right: 60px !important;
        bottom: 11px !important;
        pointer-events: none;
    }

    /* ── Predict button ── */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #ff3333 0%, #cc0000 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px;
        font-weight: 700 !important;
        font-size: 1.05rem;
        height: 54px;
        letter-spacing: 0.5px;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(255, 107, 107, 0.25);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(255, 107, 107, 0.4) !important;
    }
    div.stButton > button:first-child:active {
        transform: translateY(0);
    }

    /* ── Calories result card ── */
    .calories-result {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.12) 0%, rgba(255, 142, 83, 0.08) 100%);
        border: 1px solid rgba(255, 107, 107, 0.20);
        border-radius: 20px;
        padding: 30px 20px;
        text-align: center;
        margin-top: 10px;
        position: relative;
        overflow: hidden;
        animation: fadeSlideUp 0.5s ease-out;
    }
    .calories-result::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #ff6b6b, #ff8e53, #ffd93d);
    }
    .cal-label {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #cc9999 !important;
        margin-bottom: 6px;
    }
    .cal-value {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ff6b6b, #ff8e53, #ffd93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 4px 0;
    }
    .cal-unit {
        font-size: 0.9rem !important;
        color: #aa8888 !important;
    }

    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.06) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

    /* ── Footer ── */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 16px;
        font-size: 0.75rem !important;
        color: #555577 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load('./Model/rf_model.sav')
    return model

model = load_model()

def calculate_bmr(gender, weight, height, age):
    if gender == 'Male':
        return (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        return (10 * weight) + (6.25 * height) - (5 * age) - 161

st.markdown('<div class="hero-title"><h1>Calories Burnt Predictor</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Enter your details below to estimate calories burned during exercise</p>', unsafe_allow_html=True)

st.markdown("""
<div class="card-header">
    <span class="icon">👤</span>
    <span class="label">Personal Information</span>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    gender_input = st.selectbox("Gender", options=['Male', 'Female'])
    age_input = st.number_input("Age (years)", min_value=10, max_value=100, value=25, step=1)
with col2:
    height_input = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1, format="%.1f")
    weight_input = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.1, format="%.1f")

bmr_value = calculate_bmr(gender_input, weight_input, height_input, age_input)

st.markdown(f"""
<div style="margin-top: 8px; margin-bottom: 16px; font-size: 0.95rem; color: #a5a5cc;">
    <span style="color: #e0e0f0; font-weight: 600;">Auto-Calculated BMR: </span> {bmr_value:.1f} kcal/day
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

st.markdown("""
<div class="card-header">
    <span class="icon">🏃</span>
    <span class="label">Exercise Details</span>
</div>
""", unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    duration_input = st.number_input("Duration (minutes)", min_value=1.0, max_value=300.0, value=30.0, step=0.1, format="%.1f")
with col4:
    heart_rate_input = st.number_input("Avg Heart Rate (bpm)", min_value=60.0, max_value=200.0, value=100.0, step=0.1, format="%.1f")

st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

if st.button("Predict Calories Burned", use_container_width=True):
    input_data = pd.DataFrame({
        'Age': [age_input],
        'Heart_Rate': [heart_rate_input],
        'Duration': [duration_input],
        'BMR': [bmr_value]
    })

    predicted_calories = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="calories-result">
        <p class="cal-label">Estimated Calories Burned</p>
        <p class="cal-value">{predicted_calories:,.1f}</p>
        <p class="cal-unit">kcal</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<p class="footer">Built with Streamlit · Random Forest Regressor · Mifflin-St Jeor BMR</p>', unsafe_allow_html=True)