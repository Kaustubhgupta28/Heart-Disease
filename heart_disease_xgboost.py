import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
import streamlit.components.v1 as components
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide", initial_sidebar_state="collapsed")

# Inject particles via iframe - this ALWAYS works in Streamlit
PARTICLES_IFRAME = """
<iframe srcdoc='
<!DOCTYPE html>
<html>
<head>
<style>
* { margin:0; padding:0; overflow:hidden; }
body { background:transparent; width:100vw; height:100vh; }
.p { position:fixed; bottom:-60px; font-size:20px; pointer-events:none; animation:rise linear infinite; }
@keyframes rise {
  0%   { transform:translateY(0) rotate(0deg) scale(1);   opacity:0; }
  10%  { opacity:1; }
  85%  { opacity:0.8; }
  100% { transform:translateY(-105vh) rotate(360deg) scale(0.4); opacity:0; }
}
</style>
</head>
<body>
<script>
const emojis = ["❤️","🩷","💗","💕","🌹","🌸","🌺","🌷","💐"];
function spawn() {
  const el = document.createElement("div");
  el.className = "p";
  el.innerText = emojis[Math.floor(Math.random()*emojis.length)];
  el.style.left = (Math.random()*100) + "vw";
  el.style.fontSize = (12 + Math.random()*18) + "px";
  const dur = 4 + Math.random()*7;
  el.style.animationDuration = dur + "s";
  el.style.animationDelay = (Math.random()*2) + "s";
  document.body.appendChild(el);
  setTimeout(() => el.remove(), (dur+2)*1000);
}
setInterval(spawn, 350);
for(let i=0;i<15;i++) setTimeout(spawn, i*200);
</script>
</body>
</html>
' style="position:fixed;top:0;left:0;width:100%;height:100%;border:none;pointer-events:none;z-index:99999;background:transparent;" allowtransparency="true"></iframe>
"""

CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
[data-testid="stToolbar"] {display: none;}
.stApp > header {display: none;}
.stApp {background-color: #fff0f0 !important;}
.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}
.hero {
    background: linear-gradient(135deg, #7b0020, #9b0030, #6d001c) !important;
    padding: clamp(30px, 5vw, 60px) clamp(20px, 6vw, 80px) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    min-height: clamp(200px, 30vw, 300px) !important;
    position: relative !important;
    overflow: hidden !important;
    flex-wrap: wrap !important;
    gap: 20px !important;
}
.hero h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: clamp(2rem, 5vw, 3.5rem) !important;
    font-weight: 900 !important;
    color: white !important;
    line-height: 1.1 !important;
    margin: 0 0 12px 0 !important;
}
.hero p {
    font-family: 'Inter', sans-serif !important;
    color: rgba(255,255,255,0.6) !important;
    font-size: clamp(0.8rem, 1.5vw, 1rem) !important;
    font-weight: 300 !important;
    line-height: 1.7 !important;
    margin: 0 !important;
}
.hero-heart {
    animation: hb 2s ease-in-out infinite !important;
    flex-shrink: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
@keyframes hb { 0%,100%{transform:scale(1)} 50%{transform:scale(1.08)} }
.vdot { position: absolute !important; border-radius: 50% !important; background: radial-gradient(circle at 30% 30%, rgba(255,150,150,0.5), rgba(180,30,60,0.7)) !important; }
.vd1{width:55px;height:55px;top:15%;right:8%;}
.vd2{width:38px;height:38px;bottom:15%;right:4%;}
.vd3{width:25px;height:25px;top:20%;right:28%;}
.wrap {padding: clamp(20px, 4vw, 40px) clamp(16px, 6vw, 80px) 60px !important; position: relative !important;}
.sec-head { display: flex !important; align-items: center !important; gap: 14px !important; margin: 28px 0 14px !important; flex-wrap: wrap !important; }
.sec-icon { width: 40px !important; height: 40px !important; border-radius: 10px !important; display: flex !important; align-items: center !important; justify-content: center !important; font-size: 1.1rem !important; flex-shrink: 0 !important; }
.ic-red    { background: linear-gradient(135deg,#9b0030,#c1121f) !important; }
.ic-blue   { background: linear-gradient(135deg,#1d4ed8,#3b82f6) !important; }
.ic-green  { background: linear-gradient(135deg,#15803d,#22c55e) !important; }
.ic-orange { background: linear-gradient(135deg,#c2410c,#f97316) !important; }
.ic-purple { background: linear-gradient(135deg,#7c3aed,#a78bfa) !important; }
.sec-title { font-family:'Outfit',sans-serif !important; font-size:clamp(0.95rem,2vw,1.1rem) !important; font-weight:700 !important; color:#111 !important; margin:0 !important; }
.sec-sub   { font-size:0.75rem !important; color:#888 !important; margin:2px 0 0 !important; }
.card { background: white !important; border-radius: 16px !important; padding: clamp(16px, 3vw, 28px) !important; border: 1px solid #e5e7eb !important; box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important; margin-bottom: 8px !important; }
div[data-testid="stNumberInput"] input { background: #f9fafb !important; border: 1.5px solid #e5e7eb !important; border-radius: 10px !important; color: #111 !important; }
div[data-testid="stSelectbox"] > div > div { background: #f9fafb !important; border: 1.5px solid #e5e7eb !important; border-radius: 10px !important; }
div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label { font-size: 0.72rem !important; font-weight: 600 !important; color: #666 !important; text-transform: uppercase !important; letter-spacing: 0.04em !important; }
div[data-testid="stButton"] > button { background: linear-gradient(135deg, #9b0030, #c1121f) !important; color: white !important; border: none !important; border-radius: 14px !important; padding: 16px 40px !important; font-family: 'Outfit', sans-serif !important; font-size: 1rem !important; font-weight: 700 !important; letter-spacing: 0.06em !important; width: 100% !important; box-shadow: 0 4px 20px rgba(155,0,48,0.4) !important; }
div[data-testid="stButton"] > button:hover { transform: translateY(-2px) !important; }
.res-high { background: linear-gradient(135deg,#7b0020,#c1121f) !important; border-radius: 20px !important; padding: clamp(24px,4vw,40px) !important; text-align: center !important; color: white !important; box-shadow: 0 12px 40px rgba(155,0,48,0.4) !important; margin: 20px 0 !important; }
.res-low  { background: linear-gradient(135deg,#14532d,#15803d) !important; border-radius: 20px !important; padding: clamp(24px,4vw,40px) !important; text-align: center !important; color: white !important; box-shadow: 0 12px 40px rgba(21,128,61,0.38) !important; margin: 20px 0 !important; }
.r-emoji { font-size:clamp(2rem,5vw,3rem); display:block; margin-bottom:10px; }
.r-title { font-family:'Outfit',sans-serif; font-size:clamp(1.3rem,3vw,2rem); font-weight:900; margin:0 0 6px; }
.r-pct   { font-family:'Outfit',sans-serif; font-size:clamp(3rem,8vw,5rem); font-weight:900; line-height:1; margin:10px 0; }
.r-cap   { font-size:0.85rem; opacity:0.6; }
.r-track { height:10px; background:rgba(255,255,255,0.15); border-radius:999px; overflow:hidden; margin:18px auto; max-width:360px; }
.r-fill  { height:100%; border-radius:999px; background:linear-gradient(90deg,rgba(255,255,255,0.4),white); }
.r-pill  { display:inline-block; border-radius:999px; padding:6px 20px; font-weight:700; font-size:0.78rem; letter-spacing:0.1em; margin-top:14px; background:rgba(255,255,255,0.15); color:white; border:1px solid rgba(255,255,255,0.3); }
.causes-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 4px; }
.cause-card  { background: white; border-radius: 14px; padding: 18px; border: 1px solid #e5e7eb; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
.cause-icon  { font-size: 1.8rem; margin-bottom: 8px; display: block; }
.cause-title { font-family:'Outfit',sans-serif; font-size:0.9rem; font-weight:700; color:#1a1a2e; margin-bottom:5px; }
.cause-desc  { font-size:0.76rem; color:#6b7280; line-height:1.6; }
.sug-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin-top: 4px; }
.sug-card  { border-radius: 14px; padding: 16px 18px; border-left: 4px solid; }
.sug-card.red    { background:#fff5f5; border-color:#e63946; }
.sug-card.orange { background:#fff7ed; border-color:#f97316; }
.sug-card.green  { background:#f0fdf4; border-color:#22c55e; }
.sug-card.blue   { background:#eff6ff; border-color:#3b82f6; }
.sug-card.purple { background:#f5f3ff; border-color:#8b5cf6; }
.sug-card.teal   { background:#f0fdfa; border-color:#14b8a6; }
.sug-title { font-family:'Outfit',sans-serif; font-size:0.88rem; font-weight:700; margin-bottom:5px; color:#1a1a2e; }
.sug-text  { font-size:0.76rem; color:#4b5563; line-height:1.6; }
.warn { background:#fffbeb; border-left:4px solid #f59e0b; border-radius:10px; padding:14px 18px; font-size:0.85rem; color:#92400e; margin:12px 0; }
.foot { background:#1a1a2e; color:rgba(255,255,255,0.3); text-align:center; padding:24px; font-size:0.76rem; margin-top:60px; letter-spacing:0.03em; }
</style>
"""

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@700;800;900&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">', unsafe_allow_html=True)
st.markdown(CSS, unsafe_allow_html=True)
components.html("""
<!DOCTYPE html>
<html>
<head>
<style>
* { margin:0; padding:0; }
body { background:transparent; overflow:hidden; }
.p { position:fixed; bottom:-60px; pointer-events:none; z-index:999999; }
@keyframes rise {
  0%   { transform:translateY(0) rotate(0deg) scale(1); opacity:0; }
  10%  { opacity:1; }
  85%  { opacity:0.9; }
  100% { transform:translateY(-105vh) rotate(360deg) scale(0.3); opacity:0; }
}
</style>
</head>
<body>
<script>
const emojis = ["❤️","🩷","💗","💕","💖","🌹","🌸","🌺","🌷","💐"];
function spawn() {
  const el = document.createElement("div");
  el.className = "p";
  el.innerText = emojis[Math.floor(Math.random()*emojis.length)];
  el.style.left = (Math.random()*100) + "vw";
  el.style.fontSize = (14 + Math.random()*20) + "px";
  const dur = 5 + Math.random()*7;
  el.style.animation = "rise " + dur + "s linear forwards";
  el.style.animationDelay = (Math.random()*1) + "s";
  document.body.appendChild(el);
  setTimeout(() => el.remove(), (dur+2)*1000);
  setTimeout(spawn, 300 + Math.random()*500);
}
for(let i=0;i<12;i++) setTimeout(spawn, i*300);
</script>
</body>
</html>
""", height=0)

st.markdown("""
<div class="hero">
  <div class="vdot vd1"></div>
  <div class="vdot vd2"></div>
  <div class="vdot vd3"></div>
  <div>
    <h1>Heart Disease<br>Prediction</h1>
    <p>Analyzing risk factors to forecast likelihood of<br>developing heart conditions.</p>
  </div>
  <div class="hero-heart">
    <svg viewBox="0 0 100 90" width="220" height="200" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 8px 24px rgba(255,80,80,0.5));">
      <defs>
        <radialGradient id="hg1" cx="40%" cy="35%" r="60%">
          <stop offset="0%" stop-color="#ff6b8a"/>
          <stop offset="40%" stop-color="#e8174a"/>
          <stop offset="100%" stop-color="#8b0000"/>
        </radialGradient>
        <radialGradient id="hg2" cx="30%" cy="30%" r="40%">
          <stop offset="0%" stop-color="rgba(255,255,255,0.6)"/>
          <stop offset="100%" stop-color="rgba(255,255,255,0)"/>
        </radialGradient>
        <radialGradient id="hg3" cx="70%" cy="70%" r="40%">
          <stop offset="0%" stop-color="#6b0000"/>
          <stop offset="100%" stop-color="rgba(0,0,0,0)"/>
        </radialGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="2.5" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <path d="M50 82 C50 82 8 55 8 28 C8 14 19 5 32 5 C40 5 47 9 50 15 C53 9 60 5 68 5 C81 5 92 14 92 28 C92 55 50 82 50 82Z" fill="url(#hg1)" filter="url(#glow)"/>
      <path d="M50 82 C50 82 8 55 8 28 C8 14 19 5 32 5 C40 5 47 9 50 15 C53 9 60 5 68 5 C81 5 92 14 92 28 C92 55 50 82 50 82Z" fill="url(#hg3)" opacity="0.5"/>
      <path d="M50 82 C50 82 8 55 8 28 C8 14 19 5 32 5 C40 5 47 9 50 15 C53 9 60 5 68 5 C81 5 92 14 92 28 C92 55 50 82 50 82Z" fill="url(#hg2)"/>
      <ellipse cx="34" cy="22" rx="10" ry="7" fill="rgba(255,255,255,0.25)" transform="rotate(-25 34 22)"/>
      <path d="M46 14 C46 8 42 3 42 3 L50 3 C54 3 58 5 58 10 L58 16" stroke="#c0002a" stroke-width="5" fill="none" stroke-linecap="round"/>
      <path d="M58 10 C62 6 70 5 74 8" stroke="#3a7bd5" stroke-width="4" fill="none" stroke-linecap="round"/>
      <path d="M42 3 C36 0 28 2 26 6" stroke="#3a7bd5" stroke-width="3.5" fill="none" stroke-linecap="round"/>
      <path d="M30 35 Q28 45 32 58" stroke="rgba(180,0,40,0.4)" stroke-width="1.5" fill="none"/>
      <path d="M65 32 Q68 44 63 57" stroke="rgba(180,0,40,0.4)" stroke-width="1.5" fill="none"/>
    </svg>
  </div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    np.random.seed(42)
    n = 8000
    data = {
        'Age': np.random.randint(20,80,n),
        'Gender': np.random.choice(['Male','Female'],n),
        'BMI': np.round(np.random.uniform(16,45,n),1),
        'Sleep_Hours_Per_Day': np.random.randint(4,10,n),
        'Blood_Pressure': np.random.randint(90,180,n),
        'Cholesterol_Level': np.random.randint(150,300,n),
        'Triglyceride_Level': np.random.randint(50,400,n),
        'Fasting_Blood_Sugar': np.random.randint(70,200,n),
        'CRP_Level': np.round(np.random.uniform(0.1,10.0,n),1),
        'Homocysteine_Level': np.round(np.random.uniform(5,25,n),1),
        'Smoking': np.random.choice(['Yes','No'],n),
        'Diabetes': np.random.choice(['Yes','No'],n),
        'Family_History': np.random.choice(['Yes','No'],n),
        'High_Blood_Pressure': np.random.choice(['Yes','No'],n),
        'Low_HDL_Cholesterol': np.random.choice(['Yes','No'],n),
        'High_LDL_Cholesterol': np.random.choice(['Yes','No'],n),
        'Exercise_Habits': np.random.choice(['Regular','Occasional','None'],n),
        'Alcohol_Consumption': np.random.choice(['Heavy','Moderate','None'],n),
        'Stress_Level': np.random.choice(['High','Medium','Low'],n),
        'Sugar_Consumption': np.random.choice(['High','Medium','Low'],n),
    }
    df = pd.DataFrame(data)
    risk = (
        (df['Age']>55).astype(int)*2 +
        (df['BMI']>30).astype(int) +
        (df['Blood_Pressure']>140).astype(int)*2 +
        (df['Cholesterol_Level']>240).astype(int)*2 +
        (df['Smoking']=='Yes').astype(int)*3 +
        (df['Diabetes']=='Yes').astype(int)*2 +
        (df['Family_History']=='Yes').astype(int)*2 +
        (df['High_Blood_Pressure']=='Yes').astype(int) +
        (df['Exercise_Habits']=='None').astype(int) +
        np.random.randint(0,4,n)
    )
    df['Heart_Disease'] = (risk>=10).astype(int)
    cat_cols = ['Gender','Smoking','Diabetes','Family_History','High_Blood_Pressure',
                'Low_HDL_Cholesterol','High_LDL_Cholesterol','Exercise_Habits',
                'Alcohol_Consumption','Stress_Level','Sugar_Consumption']
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
        encoders[c] = le
    X = df.drop('Heart_Disease', axis=1)
    y = df['Heart_Disease']
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xtr, _, ytr, _ = train_test_split(Xs, y, test_size=0.2, random_state=42)
    m = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                           random_state=42, use_label_encoder=False, eval_metric='logloss')
    m.fit(Xtr, ytr)
    return m, sc, encoders, X.columns.tolist()

model, scaler, encoders, feature_cols = load_model()

st.markdown('<div class="wrap">', unsafe_allow_html=True)

st.markdown("""
<div class="sec-head">
  <div class="sec-icon ic-orange">⚠️</div>
  <div>
    <p class="sec-title">Common Causes of Heart Disease</p>
    <p class="sec-sub">Understanding the key risk factors that contribute to heart conditions</p>
  </div>
</div>
<div class="card">
  <div class="causes-grid">
    <div class="cause-card"><span class="cause-icon">🚬</span><p class="cause-title">Smoking</p><p class="cause-desc">Tobacco damages blood vessel walls, reduces oxygen in blood, and raises blood pressure significantly.</p></div>
    <div class="cause-card"><span class="cause-icon">🩸</span><p class="cause-title">High Blood Pressure</p><p class="cause-desc">Hypertension forces the heart to work harder, thickening the heart muscle and hardening arteries over time.</p></div>
    <div class="cause-card"><span class="cause-icon">🍔</span><p class="cause-title">High Cholesterol</p><p class="cause-desc">Excess LDL cholesterol builds up as plaque inside arteries, narrowing them and restricting blood flow.</p></div>
    <div class="cause-card"><span class="cause-icon">🍩</span><p class="cause-title">Diabetes</p><p class="cause-desc">High blood sugar damages blood vessels and nerves controlling the heart, doubling cardiovascular risk.</p></div>
    <div class="cause-card"><span class="cause-icon">😰</span><p class="cause-title">Chronic Stress</p><p class="cause-desc">Ongoing stress raises cortisol and adrenaline, increasing blood pressure and promoting artery inflammation.</p></div>
    <div class="cause-card"><span class="cause-icon">🛋️</span><p class="cause-title">Physical Inactivity</p><p class="cause-desc">A sedentary lifestyle leads to obesity, high blood pressure, and poor cholesterol levels.</p></div>
    <div class="cause-card"><span class="cause-icon">🧬</span><p class="cause-title">Family History</p><p class="cause-desc">Genetics play a major role. If a close relative had heart disease, your risk is significantly higher.</p></div>
    <div class="cause-card"><span class="cause-icon">⚖️</span><p class="cause-title">Obesity</p><p class="cause-desc">Excess body weight strains the heart, raises blood pressure, and is linked to diabetes and high cholesterol.</p></div>
    <div class="cause-card"><span class="cause-icon">🍺</span><p class="cause-title">Excess Alcohol</p><p class="cause-desc">Heavy drinking weakens the heart muscle, raises blood pressure, and contributes to irregular heartbeat.</p></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sec-head">
  <div class="sec-icon ic-red">👤</div>
  <div>
    <p class="sec-title">Personal Information</p>
    <p class="sec-sub">Basic demographic and physical details</p>
  </div>
</div>
<div class="card">
""", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1: age = st.number_input("AGE (YEARS)", min_value=0, max_value=120, value=0)
with c2: gender = st.selectbox("GENDER", ["Select","Male","Female"])
with c3: bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0, step=0.1)
with c4: sleep = st.number_input("SLEEP HOURS/DAY", min_value=0, max_value=24, value=0)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="sec-head">
  <div class="sec-icon ic-blue">🔬</div>
  <div>
    <p class="sec-title">Clinical Measurements</p>
    <p class="sec-sub">Lab results and vital signs</p>
  </div>
</div>
<div class="card">
""", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    bp   = st.number_input("BLOOD PRESSURE (mmHg)", min_value=0, max_value=300, value=0)
    fbs  = st.number_input("FASTING BLOOD SUGAR (mg/dL)", min_value=0, max_value=500, value=0)
with c2:
    chol = st.number_input("CHOLESTEROL (mg/dL)", min_value=0, max_value=600, value=0)
    crp  = st.number_input("CRP LEVEL (mg/L)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
with c3:
    trig = st.number_input("TRIGLYCERIDE (mg/dL)", min_value=0, max_value=1000, value=0)
    homo = st.number_input("HOMOCYSTEINE (µmol/L)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="sec-head">
  <div class="sec-icon ic-green">🏃</div>
  <div>
    <p class="sec-title">Health &amp; Lifestyle</p>
    <p class="sec-sub">Medical history and daily habits</p>
  </div>
</div>
<div class="card">
""", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    r1, r2 = st.columns(2)
    with r1:
        smoking  = st.selectbox("SMOKING",       ["Select","Yes","No"])
        fam_hist = st.selectbox("FAMILY HISTORY", ["Select","Yes","No"])
        low_hdl  = st.selectbox("LOW HDL",        ["Select","Yes","No"])
        exercise = st.selectbox("EXERCISE",       ["Select","Regular","Occasional","None"])
        stress   = st.selectbox("STRESS LEVEL",   ["Select","High","Medium","Low"])
    with r2:
        diabetes = st.selectbox("DIABETES",       ["Select","Yes","No"])
        high_bp  = st.selectbox("HIGH BP",        ["Select","Yes","No"])
        high_ldl = st.selectbox("HIGH LDL",       ["Select","Yes","No"])
        alcohol  = st.selectbox("ALCOHOL",        ["Select","Heavy","Moderate","None"])
        sugar    = st.selectbox("SUGAR INTAKE",   ["Select","High","Medium","Low"])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1,2,1])
with btn_col:
    predict = st.button("❤️  ANALYZE HEART DISEASE RISK")

if predict:
    selects = [gender, smoking, diabetes, fam_hist, high_bp, low_hdl, high_ldl, exercise, alcohol, stress, sugar]
    nums    = [age, bmi, bp, chol, trig, fbs, crp, homo, sleep]

    if any(s == "Select" for s in selects) or any(v == 0 for v in nums):
        st.markdown('<div class="warn">⚠️ <strong>Incomplete form.</strong> Please fill in all fields before predicting.</div>', unsafe_allow_html=True)
    else:
        def enc(col, val):
            try: return int(encoders[col].transform([val])[0])
            except: return 0

        inp = {
            'Age': age, 'Gender': enc('Gender', gender), 'BMI': bmi,
            'Sleep_Hours_Per_Day': sleep, 'Blood_Pressure': bp,
            'Cholesterol_Level': chol, 'Triglyceride_Level': trig,
            'Fasting_Blood_Sugar': fbs, 'CRP_Level': crp, 'Homocysteine_Level': homo,
            'Smoking': enc('Smoking', smoking), 'Diabetes': enc('Diabetes', diabetes),
            'Family_History': enc('Family_History', fam_hist),
            'High_Blood_Pressure': enc('High_Blood_Pressure', high_bp),
            'Low_HDL_Cholesterol': enc('Low_HDL_Cholesterol', low_hdl),
            'High_LDL_Cholesterol': enc('High_LDL_Cholesterol', high_ldl),
            'Exercise_Habits': enc('Exercise_Habits', exercise),
            'Alcohol_Consumption': enc('Alcohol_Consumption', alcohol),
            'Stress_Level': enc('Stress_Level', stress),
            'Sugar_Consumption': enc('Sugar_Consumption', sugar),
        }
        df_in    = pd.DataFrame([inp])[feature_cols]
        scaled   = scaler.transform(df_in)
        prob     = float(model.predict_proba(scaled)[0][1])
        pred_val = int(model.predict(scaled)[0])
        pct      = round(prob * 100, 1)

        if pred_val == 1:
            level = "🔴 HIGH RISK" if pct >= 70 else "⚡ MEDIUM-HIGH RISK"
            st.markdown(f'<div class="res-high"><span class="r-emoji">🔥</span><p class="r-title">HIGH RISK DETECTED</p><p class="r-pct">{pct}%</p><p class="r-cap">Probability of developing heart disease</p><div class="r-track"><div class="r-fill" style="width:{pct}%"></div></div><span class="r-pill">{level}</span></div>', unsafe_allow_html=True)
        else:
            level = "✔ LOW RISK" if pct < 30 else "⚡ LOW-MEDIUM RISK"
            st.markdown(f'<div class="res-low"><span class="r-emoji">✅</span><p class="r-title">LOW RISK</p><p class="r-pct">{pct}%</p><p class="r-cap">Probability of developing heart disease</p><div class="r-track"><div class="r-fill" style="width:{pct}%"></div></div><span class="r-pill">{level}</span></div>', unsafe_allow_html=True)

        suggestions = []
        if smoking == "Yes":
            suggestions.append(("red","🚬","Quit Smoking Immediately","Smoking is the #1 modifiable risk factor. Quitting within 1 year halves your heart disease risk."))
        if bp > 140 or high_bp == "Yes":
            suggestions.append(("orange","🩸","Lower Your Blood Pressure","Your BP is elevated. Reduce salt intake, avoid caffeine, exercise regularly, and take prescribed medication."))
        if chol > 240 or high_ldl == "Yes":
            suggestions.append(("orange","🥑","Improve Your Cholesterol","Cut saturated fats, eat more oats, nuts and olive oil. Ask your doctor about statins if needed."))
        if bmi > 30:
            suggestions.append(("red","⚖️","Manage Your Weight","A BMI above 30 significantly strains your heart. Aim to lose 5-10% of body weight through diet and activity."))
        if exercise == "None":
            suggestions.append(("blue","🏃","Start Exercising Regularly","Aim for at least 150 minutes of moderate activity per week. Even 30 mins/day reduces heart risk by 35%."))
        if exercise == "Occasional":
            suggestions.append(("blue","🏃","Increase Exercise Frequency","Move from occasional to regular exercise. Try to be active 5 days a week."))
        if stress == "High":
            suggestions.append(("purple","🧘","Manage Stress Actively","Chronic stress raises cortisol and damages arteries. Practice daily meditation or yoga."))
        if diabetes == "Yes" or fbs > 126:
            suggestions.append(("teal","💉","Control Blood Sugar","High blood sugar damages blood vessels. Follow a low-glycaemic diet and take medication as prescribed."))
        if alcohol == "Heavy":
            suggestions.append(("orange","🍺","Reduce Alcohol Intake","Heavy drinking weakens the heart muscle. Limit to max 1 drink/day for women and 2 for men."))
        if sleep < 6:
            suggestions.append(("blue","😴","Improve Sleep Quality","Less than 6 hours raises heart disease risk. Establish a consistent sleep schedule."))
        if low_hdl == "Yes":
            suggestions.append(("green","🥜","Boost Your HDL Cholesterol","Eat more healthy fats (avocado, nuts, olive oil) and exercise regularly."))
        if trig > 200:
            suggestions.append(("orange","🐟","Lower Triglycerides","Cut sugar, refined carbs and alcohol. Eat fatty fish twice a week."))
        if sugar == "High":
            suggestions.append(("red","🍬","Cut Down on Sugar","High sugar leads to obesity and inflammation. Replace sweets with fruits and whole grains."))
        if pred_val == 0:
            suggestions.append(("green","✅","Keep Up the Good Work!","Your results look positive! Maintain healthy habits — exercise, balanced diet and routine check-ups."))
            suggestions.append(("teal","🏥","Schedule Regular Check-ups","Even with low risk, get a heart health check every 1-2 years."))

        if suggestions:
            st.markdown('<div class="sec-head"><div class="sec-icon ic-purple">💡</div><div><p class="sec-title">Personalised Health Suggestions</p><p class="sec-sub">Based on your results — steps you can take to improve your heart health</p></div></div><div class="card"><div class="sug-grid">', unsafe_allow_html=True)
            for color, icon, title, text in suggestions:
                st.markdown(f'<div class="sug-card {color}"><p class="sug-title">{icon} {title}</p><p class="sug-text">{text}</p></div>', unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

        st.markdown('<p style="font-size:0.74rem;color:#9ca3af;text-align:center;margin-top:16px;">⚕️ This tool can help you understand your heart health — but always consult a qualified doctor before making any medical decisions.</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="foot">❤️ Heart Disease Prediction &nbsp;·&nbsp; Powered by XGBoost &nbsp;·&nbsp; For educational use only</div>', unsafe_allow_html=True)
