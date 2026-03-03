import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Heart Risk Analyser", page_icon="❤️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }

.stApp { background: #0a0600; color: #e2d5c8; }

.hero {
    background: linear-gradient(135deg, #1a0800 0%, #2d1200 50%, #1a0800 100%);
    border: 1px solid rgba(251,146,60,0.15);
    border-radius: 24px; padding: 48px 44px;
    margin-bottom: 28px; position: relative; overflow: hidden;
}
.hero::before {
    content: '❤';
    position: absolute; right: 48px; top: 50%;
    transform: translateY(-50%);
    font-size: 8rem; opacity: 0.04; line-height: 1;
}
.hero::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(circle at 20% 50%, rgba(251,146,60,0.07) 0%, transparent 60%),
                radial-gradient(circle at 80% 50%, rgba(239,68,68,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-badge {
    font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: #fb923c; margin-bottom: 12px; font-weight: 600;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #fb923c, #f97316, #ef4444);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px; line-height: 1.1;
}
.hero-sub { color: #6b4a2a; font-size: 0.95rem; font-weight: 300; }

.card {
    background: #110800;
    border: 1px solid rgba(251,146,60,0.1);
    border-radius: 18px; padding: 28px; margin-bottom: 20px;
    transition: border-color 0.3s;
}
.card:hover { border-color: rgba(251,146,60,0.25); }

.card-title {
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #fb923c; margin-bottom: 20px;
    display: flex; align-items: center; gap: 8px;
}
.card-title::after { content: ''; flex: 1; height: 1px; background: rgba(251,146,60,0.1); }

.stNumberInput input {
    background: #1a0e04 !important;
    border: 1px solid rgba(251,146,60,0.15) !important;
    border-radius: 10px !important;
    color: #d4a06a !important;
    font-size: 0.9rem !important;
}
.stNumberInput input:focus {
    border-color: rgba(251,146,60,0.5) !important;
    box-shadow: 0 0 0 3px rgba(251,146,60,0.08) !important;
    background: #1f1004 !important;
}
div[data-baseweb="select"] > div {
    background: #1a0e04 !important;
    border: 1px solid rgba(251,146,60,0.15) !important;
    border-radius: 10px !important;
    color: #d4a06a !important;
}
label { color: #7a5a3a !important; font-size: 0.82rem !important; }

.stButton button {
    background: linear-gradient(135deg, #ea580c, #dc2626) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 16px 32px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important; font-size: 1rem !important;
    letter-spacing: 0.05em !important; width: 100% !important;
    transition: all 0.3s !important;
}
.stButton button:hover {
    box-shadow: 0 12px 36px rgba(234,88,12,0.4) !important;
    transform: translateY(-2px) !important;
}

.result-high {
    background: linear-gradient(135deg, #1f0500, #2d0a00);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 22px; padding: 44px; text-align: center;
}
.result-low {
    background: linear-gradient(135deg, #001f08, #002d0f);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 22px; padding: 44px; text-align: center;
}
.result-icon { font-size: 4rem; margin-bottom: 12px; }
.result-title { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; margin-bottom: 6px; }
.result-pct { font-family: 'Syne', sans-serif; font-size: 4.5rem; font-weight: 800; line-height: 1; margin: 10px 0 6px; }
.result-lbl { font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 14px; }
.risk-pill { display: inline-block; border-radius: 6px; padding: 5px 18px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; }
.risk-high { background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.3); color: #f87171; }
.risk-medium { background: rgba(251,146,60,0.1); border: 1px solid rgba(251,146,60,0.3); color: #fb923c; }
.risk-low { background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3); color: #4ade80; }

.stProgress > div > div {
    background: linear-gradient(90deg, #fb923c, #f97316, #ef4444) !important;
    border-radius: 999px !important;
}
.stProgress > div { background: rgba(255,255,255,0.04) !important; border-radius: 999px !important; }

.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(251,146,60,0.2), transparent); margin: 28px 0; }
footer { display: none; }
#MainMenu { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Train Model ───────────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv("heart_disease.csv")
    df['Alcohol Consumption'] = df['Alcohol Consumption'].fillna('Unknown')
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    df['Heart Disease Status'] = (df['Heart Disease Status'] == 'Yes').astype(int)
    df['Gender'] = (df['Gender'] == 'Male').astype(int)
    for col in ['Smoking','Family Heart Disease','Diabetes','High Blood Pressure','Low HDL Cholesterol','High LDL Cholesterol']:
        df[col] = (df[col] == 'Yes').astype(int)
    for col in ['Exercise Habits','Stress Level','Sugar Consumption']:
        df[col] = df[col].map({'Low':0,'Medium':1,'High':2})
    df['Alcohol Consumption'] = df['Alcohol Consumption'].map({'Unknown':-1,'Low':0,'Medium':1,'High':2})
    df['BMI_BP_ratio']         = df['BMI'] / (df['Blood Pressure'] + 1)
    df['Lipid_risk_score']     = df['Cholesterol Level'] + df['Triglyceride Level']
    df['Metabolic_risk_score'] = df['Fasting Blood Sugar'] + df['CRP Level'] + df['Homocysteine Level']
    clip_cols = ['Blood Pressure','Cholesterol Level','BMI','Triglyceride Level','Fasting Blood Sugar','CRP Level','Homocysteine Level','Sleep Hours','Age']
    clip_bounds = {}
    for col in clip_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        clip_bounds[col] = (Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))
        df[col] = df[col].clip(*clip_bounds[col])
    scale_cols = ['Age','Blood Pressure','Cholesterol Level','BMI','Sleep Hours','Triglyceride Level','Fasting Blood Sugar','CRP Level','Homocysteine Level','BMI_BP_ratio','Lipid_risk_score','Metabolic_risk_score']
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    feature_columns = [c for c in df.columns if c != 'Heart Disease Status']
    X, y = df[feature_columns], df['Heart Disease Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=4.0, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model, scaler, clip_bounds, feature_columns

with st.spinner("🔥 Warming up the engine..."):
    model, scaler, clip_bounds, feature_columns = train_model()

# ── Hero ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">▸ AI-Powered Risk Assessment</div>
    <div class="hero-title">❤️ Heart Risk Analyser</div>
    <div class="hero-sub">Fill in your health details below for an instant cardiovascular risk prediction</div>
</div>
""", unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card"><div class="card-title">👤 Personal Info</div>', unsafe_allow_html=True)
    age    = st.number_input("Age (years)",         min_value=0,   max_value=100, value=0)
    gender = st.selectbox("Gender",                 ["Select","Male","Female"])
    bmi    = st.number_input("BMI",                 min_value=0.0, max_value=60.0, value=0.0)
    sleep  = st.number_input("Sleep Hours / night", min_value=0.0, max_value=12.0, value=0.0)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-title">🩺 Clinical Measurements</div>', unsafe_allow_html=True)
    bp           = st.number_input("Blood Pressure (mmHg)",       min_value=0,   max_value=250, value=0)
    cholesterol  = st.number_input("Cholesterol Level (mg/dL)",   min_value=0,   max_value=400, value=0)
    triglyceride = st.number_input("Triglyceride Level (mg/dL)",  min_value=0,   max_value=600, value=0)
    fbs          = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=0,   max_value=300, value=0)
    crp          = st.number_input("CRP Level (mg/L)",            min_value=0.0, max_value=20.0, value=0.0)
    homo         = st.number_input("Homocysteine (μmol/L)",       min_value=0.0, max_value=30.0, value=0.0)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card"><div class="card-title">🧬 Health & Lifestyle</div>', unsafe_allow_html=True)
    smoking   = st.selectbox("Smoking",              ["Select","No","Yes"])
    diabetes  = st.selectbox("Diabetes",             ["Select","No","Yes"])
    family_hd = st.selectbox("Family Heart Disease", ["Select","No","Yes"])
    high_bp   = st.selectbox("High Blood Pressure",  ["Select","No","Yes"])
    low_hdl   = st.selectbox("Low HDL Cholesterol",  ["Select","No","Yes"])
    high_ldl  = st.selectbox("High LDL Cholesterol", ["Select","No","Yes"])
    exercise  = st.selectbox("Exercise Habits",      ["Select","Low","Medium","High"])
    alcohol   = st.selectbox("Alcohol Consumption",  ["Select","Unknown","Low","Medium","High"])
    stress    = st.selectbox("Stress Level",         ["Select","Low","Medium","High"])
    sugar     = st.selectbox("Sugar Consumption",    ["Select","Low","Medium","High"])
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────
if st.button("⚡ Analyse My Heart Risk", use_container_width=True):
    dropdowns = [gender, smoking, diabetes, family_hd, high_bp, low_hdl, high_ldl, exercise, alcohol, stress, sugar]
    if "Select" in dropdowns or age == 0 or bp == 0 or cholesterol == 0:
        st.warning("⚠️ Please fill in all fields before running the analysis.")
    else:
        mp = pd.DataFrame([{
            'Age':age,'Gender':gender,'Blood Pressure':bp,'Cholesterol Level':cholesterol,
            'Exercise Habits':exercise,'Smoking':smoking,'Family Heart Disease':family_hd,
            'Diabetes':diabetes,'BMI':bmi,'High Blood Pressure':high_bp,
            'Low HDL Cholesterol':low_hdl,'High LDL Cholesterol':high_ldl,
            'Alcohol Consumption':alcohol,'Stress Level':stress,'Sleep Hours':sleep,
            'Sugar Consumption':sugar,'Triglyceride Level':triglyceride,
            'Fasting Blood Sugar':fbs,'CRP Level':crp,'Homocysteine Level':homo
        }])
        mp['Gender'] = (mp['Gender']=='Male').astype(int)
        for col in ['Smoking','Family Heart Disease','Diabetes','High Blood Pressure','Low HDL Cholesterol','High LDL Cholesterol']:
            mp[col] = (mp[col]=='Yes').astype(int)
        for col in ['Exercise Habits','Stress Level','Sugar Consumption']:
            mp[col] = mp[col].map({'Low':0,'Medium':1,'High':2})
        mp['Alcohol Consumption'] = mp['Alcohol Consumption'].map({'Unknown':-1,'Low':0,'Medium':1,'High':2})
        mp['BMI_BP_ratio']         = mp['BMI'] / (mp['Blood Pressure'] + 1)
        mp['Lipid_risk_score']     = mp['Cholesterol Level'] + mp['Triglyceride Level']
        mp['Metabolic_risk_score'] = mp['Fasting Blood Sugar'] + mp['CRP Level'] + mp['Homocysteine Level']
        for col in ['Blood Pressure','Cholesterol Level','BMI','Triglyceride Level','Fasting Blood Sugar','CRP Level','Homocysteine Level','Sleep Hours','Age']:
            mp[col] = mp[col].clip(*clip_bounds[col])
        scale_cols = ['Age','Blood Pressure','Cholesterol Level','BMI','Sleep Hours','Triglyceride Level','Fasting Blood Sugar','CRP Level','Homocysteine Level','BMI_BP_ratio','Lipid_risk_score','Metabolic_risk_score']
        mp[scale_cols] = scaler.transform(mp[scale_cols])

        prob = model.predict_proba(mp[feature_columns])[0][1]
        pred = model.predict(mp[feature_columns])[0]

        _, rc, _ = st.columns([1, 2, 1])
        with rc:
            if pred == 1:
                pill = '<span class="risk-pill risk-high">⚠ HIGH RISK</span>' if prob > 0.6 else '<span class="risk-pill risk-medium">⚡ MEDIUM RISK</span>'
                st.markdown(f"""
                <div class="result-high">
                    <div class="result-icon">🚨</div>
                    <div class="result-title" style="color:#f87171">High Risk Detected</div>
                    <div class="result-pct" style="background:linear-gradient(135deg,#fb923c,#ef4444);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{prob*100:.1f}%</div>
                    <div class="result-lbl" style="color:#7a3a2a">Probability of Heart Disease</div>
                    {pill}
                </div>""", unsafe_allow_html=True)
            else:
                pill = '<span class="risk-pill risk-low">✔ LOW RISK</span>' if prob < 0.3 else '<span class="risk-pill risk-medium">⚡ MEDIUM RISK</span>'
                st.markdown(f"""
                <div class="result-low">
                    <div class="result-icon">✅</div>
                    <div class="result-title" style="color:#4ade80">You're Looking Good!</div>
                    <div class="result-pct" style="color:#22c55e">{prob*100:.1f}%</div>
                    <div class="result-lbl" style="color:#2a5a3a">Probability of Heart Disease</div>
                    {pill}
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(float(prob))

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#4a2a1a;font-size:0.78rem">⚠️ For educational purposes only. Always consult a qualified medical professional.</p>', unsafe_allow_html=True)
