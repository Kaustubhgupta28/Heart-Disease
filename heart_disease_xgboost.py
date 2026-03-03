import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
#MainMenu,footer,header,.stDeployButton{display:none!important}
.stApp{background:#f0f2f5;font-family:'Inter',sans-serif}
[data-testid="stAppViewContainer"]>.main{padding:0!important}
.block-container{padding:0!important;max-width:100%!important}
.hero-section{background:linear-gradient(135deg,#7b0020 0%,#9b0030 35%,#8b0025 65%,#6d001c 100%);padding:0;margin:0;position:relative;overflow:hidden;min-height:320px;display:flex;align-items:center}
.hero-section::before{content:'';position:absolute;width:550px;height:550px;border-radius:50%;background:rgba(255,255,255,0.03);top:-200px;right:-80px;z-index:1}
.hero-section::after{content:'';position:absolute;width:280px;height:280px;border-radius:50%;background:rgba(255,255,255,0.04);bottom:-100px;right:220px;z-index:1}
.hero-inner{width:100%;padding:55px 70px;display:flex;align-items:center;justify-content:space-between;position:relative;z-index:2}
.hero-left{max-width:480px}
.hero-title{font-family:'Outfit',sans-serif;font-size:3.6rem;font-weight:900;color:#fff;line-height:1.08;margin:0 0 18px 0;letter-spacing:-1px}
.hero-subtitle{font-family:'Inter',sans-serif;font-size:0.98rem;color:rgba(255,255,255,0.6);font-weight:300;line-height:1.75;margin:0}
.virus-dot{position:absolute;border-radius:50%;background:radial-gradient(circle at 30% 30%,rgba(255,160,160,0.55),rgba(180,30,60,0.75));z-index:2}
.virus-dot-1{width:58px;height:58px;top:14%;right:9%}
.virus-dot-2{width:40px;height:40px;bottom:16%;right:5%}
.virus-dot-3{width:26px;height:26px;top:22%;right:30%}
.hero-heart{font-size:8rem;filter:drop-shadow(0 24px 48px rgba(0,0,0,0.45));animation:heartbeat 2.2s ease-in-out infinite;position:relative;z-index:2;margin-right:30px}
@keyframes heartbeat{0%,100%{transform:scale(1)}50%{transform:scale(1.07)}}
.content-wrap{padding:40px 70px 60px;max-width:1400px;margin:0 auto}
.section-head{display:flex;align-items:center;gap:14px;margin-bottom:18px;margin-top:34px}
.section-icon{width:40px;height:40px;border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:1.15rem;flex-shrink:0}
.icon-red{background:linear-gradient(135deg,#9b0030,#c1121f)}
.icon-blue{background:linear-gradient(135deg,#1d4ed8,#3b82f6)}
.icon-green{background:linear-gradient(135deg,#15803d,#22c55e)}
.section-title{font-family:'Outfit',sans-serif;font-size:1.1rem;font-weight:700;color:#1a1a2e;margin:0}
.section-sub{font-size:0.76rem;color:#6b7280;margin:2px 0 0}
.card{background:white;border-radius:16px;padding:28px;border:1px solid #e5e7eb;box-shadow:0 1px 4px rgba(0,0,0,0.05),0 4px 14px rgba(0,0,0,0.05);margin-bottom:4px}
div[data-testid="stNumberInput"] input,div[data-testid="stSelectbox"]>div>div{background:#f9fafb!important;border:1.5px solid #e5e7eb!important;border-radius:10px!important;font-family:'Inter',sans-serif!important;font-size:0.87rem!important;color:#1a1a2e!important}
div[data-testid="stNumberInput"] label,div[data-testid="stSelectbox"] label{font-family:'Inter',sans-serif!important;font-size:0.76rem!important;font-weight:600!important;color:#6b7280!important;letter-spacing:0.03em!important;text-transform:uppercase!important}
div[data-testid="stButton"]>button{width:100%!important;background:linear-gradient(135deg,#9b0030,#c1121f,#9b0030)!important;background-size:200%!important;color:white!important;border:none!important;border-radius:14px!important;padding:18px 40px!important;font-family:'Outfit',sans-serif!important;font-size:1.05rem!important;font-weight:700!important;letter-spacing:0.06em!important;text-transform:uppercase!important;box-shadow:0 4px 20px rgba(155,0,48,0.35)!important;transition:all 0.3s!important}
div[data-testid="stButton"]>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 28px rgba(155,0,48,0.45)!important}
.result-high{background:linear-gradient(135deg,#7b0020,#9b0030,#c1121f);border-radius:20px;padding:42px;text-align:center;color:white;box-shadow:0 14px 44px rgba(155,0,48,0.38);margin:24px 0;animation:fadeUp 0.5s ease}
.result-low{background:linear-gradient(135deg,#14532d,#166534,#15803d);border-radius:20px;padding:42px;text-align:center;color:white;box-shadow:0 14px 44px rgba(21,128,61,0.35);margin:24px 0;animation:fadeUp 0.5s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
.r-emoji{font-size:3.2rem;display:block;margin-bottom:10px}
.r-verdict{font-family:'Outfit',sans-serif;font-size:2rem;font-weight:900;margin:0 0 4px}
.r-prob{font-family:'Outfit',sans-serif;font-size:4.5rem;font-weight:900;line-height:1;margin:14px 0 6px}
.r-caption{font-size:0.85rem;opacity:0.6;font-weight:300}
.r-bar-wrap{margin:20px auto;max-width:380px}
.r-bar-labels{display:flex;justify-content:space-between;font-size:0.75rem;color:rgba(255,255,255,0.55);margin-bottom:8px}
.r-bar-track{height:10px;background:rgba(255,255,255,0.15);border-radius:999px;overflow:hidden}
.r-bar-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,rgba(255,255,255,0.45),white)}
.r-pill{display:inline-block;border-radius:999px;padding:7px 22px;font-weight:700;font-size:0.78rem;letter-spacing:0.1em;margin-top:16px;background:rgba(255,255,255,0.14);color:white;border:1px solid rgba(255,255,255,0.28)}
.warn-box{background:#fffbeb;border:1px solid #fbbf24;border-left:4px solid #f59e0b;border-radius:10px;padding:14px 18px;font-size:0.83rem;color:#92400e;margin-bottom:20px}
.app-footer{background:#1a1a2e;color:rgba(255,255,255,0.3);text-align:center;padding:26px;font-size:0.76rem;margin-top:60px;letter-spacing:0.03em}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-section">
  <div class="virus-dot virus-dot-1"></div>
  <div class="virus-dot virus-dot-2"></div>
  <div class="virus-dot virus-dot-3"></div>
  <div class="hero-inner">
    <div class="hero-left">
      <h1 class="hero-title">Heart Disease<br>Prediction</h1>
      <p class="hero-subtitle">Analyzing risk factors to forecast likelihood of<br>developing heart conditions.</p>
    </div>
    <div class="hero-heart">🫀</div>
  </div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    np.random.seed(42)
    n = 8000
    data = {
        'Age': np.random.randint(20,80,n), 'Gender': np.random.choice(['Male','Female'],n),
        'BMI': np.round(np.random.uniform(16,45,n),1), 'Sleep_Hours_Per_Day': np.random.randint(4,10,n),
        'Blood_Pressure': np.random.randint(90,180,n), 'Cholesterol_Level': np.random.randint(150,300,n),
        'Triglyceride_Level': np.random.randint(50,400,n), 'Fasting_Blood_Sugar': np.random.randint(70,200,n),
        'CRP_Level': np.round(np.random.uniform(0.1,10.0,n),1), 'Homocysteine_Level': np.round(np.random.uniform(5,25,n),1),
        'Smoking': np.random.choice(['Yes','No'],n), 'Diabetes': np.random.choice(['Yes','No'],n),
        'Family_History': np.random.choice(['Yes','No'],n), 'High_Blood_Pressure': np.random.choice(['Yes','No'],n),
        'Low_HDL_Cholesterol': np.random.choice(['Yes','No'],n), 'High_LDL_Cholesterol': np.random.choice(['Yes','No'],n),
        'Exercise_Habits': np.random.choice(['Regular','Occasional','None'],n),
        'Alcohol_Consumption': np.random.choice(['Heavy','Moderate','None'],n),
        'Stress_Level': np.random.choice(['High','Medium','Low'],n),
        'Sugar_Consumption': np.random.choice(['High','Medium','Low'],n),
    }
    df = pd.DataFrame(data)
    risk = ((df['Age']>55).astype(int)*2+(df['BMI']>30).astype(int)+(df['Blood_Pressure']>140).astype(int)*2+
            (df['Cholesterol_Level']>240).astype(int)*2+(df['Smoking']=='Yes').astype(int)*3+
            (df['Diabetes']=='Yes').astype(int)*2+(df['Family_History']=='Yes').astype(int)*2+
            (df['High_Blood_Pressure']=='Yes').astype(int)+(df['Exercise_Habits']=='None').astype(int)+np.random.randint(0,4,n))
    df['Heart_Disease'] = (risk>=10).astype(int)
    cat_cols = ['Gender','Smoking','Diabetes','Family_History','High_Blood_Pressure','Low_HDL_Cholesterol',
                'High_LDL_Cholesterol','Exercise_Habits','Alcohol_Consumption','Stress_Level','Sugar_Consumption']
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
        encoders[c] = le
    X = df.drop('Heart_Disease', axis=1); y = df['Heart_Disease']
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42)
    model = xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,random_state=42,use_label_encoder=False,eval_metric='logloss')
    model.fit(Xtr,ytr)
    return model, scaler, encoders, X.columns.tolist()

model, scaler, encoders, feature_cols = load_model()

st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

st.markdown("""<div class="section-head"><div class="section-icon icon-red">👤</div>
<div><p class="section-title">Personal Information</p><p class="section-sub">Basic demographic and physical details</p></div></div>""", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
with c1: age = st.number_input("Age (years)", min_value=1, max_value=120, value=0)
with c2: gender = st.selectbox("Gender", ["Select","Male","Female"])
with c3: bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0, step=0.1)
with c4: sleep = st.number_input("Sleep Hours / Day", min_value=0, max_value=24, value=0)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""<div class="section-head"><div class="section-icon icon-blue">🔬</div>
<div><p class="section-title">Clinical Measurements</p><p class="section-sub">Lab results and vital signs</p></div></div>""", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
c1,c2,c3 = st.columns(3)
with c1:
    bp = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=300, value=0)
    fbs = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=0, max_value=500, value=0)
with c2:
    chol = st.number_input("Cholesterol Level (mg/dL)", min_value=0, max_value=600, value=0)
    crp = st.number_input("CRP Level (mg/L)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
with c3:
    trig = st.number_input("Triglyceride Level (mg/dL)", min_value=0, max_value=1000, value=0)
    homo = st.number_input("Homocysteine Level (µmol/L)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""<div class="section-head"><div class="section-icon icon-green">🏃</div>
<div><p class="section-title">Health &amp; Lifestyle</p><p class="section-sub">Medical history and daily habits</p></div></div>""", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    smoking = st.selectbox("Smoking", ["Select","Yes","No"])
    diabetes = st.selectbox("Diabetes", ["Select","Yes","No"])
with c2:
    fam_hist = st.selectbox("Family History", ["Select","Yes","No"])
    high_bp = st.selectbox("High BP", ["Select","Yes","No"])
with c3:
    low_hdl = st.selectbox("Low HDL", ["Select","Yes","No"])
    high_ldl = st.selectbox("High LDL", ["Select","Yes","No"])
with c4:
    exercise = st.selectbox("Exercise", ["Select","Regular","Occasional","None"])
    alcohol = st.selectbox("Alcohol", ["Select","Heavy","Moderate","None"])
with c5:
    stress = st.selectbox("Stress Level", ["Select","High","Medium","Low"])
    sugar = st.selectbox("Sugar Intake", ["Select","High","Medium","Low"])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
_,btn_col,_ = st.columns([1,2,1])
with btn_col:
    predict = st.button("❤️  ANALYZE HEART DISEASE RISK")

if predict:
    selects = [gender,smoking,diabetes,fam_hist,high_bp,low_hdl,high_ldl,exercise,alcohol,stress,sugar]
    nums = [age,bmi,bp,chol,trig,fbs,crp,homo,sleep]
    if any(s=="Select" for s in selects) or any(v==0 for v in nums):
        st.markdown('<div class="warn-box">⚠️ <strong>Incomplete form.</strong> Please fill in all fields before running the prediction.</div>', unsafe_allow_html=True)
    else:
        def enc(col,val):
            try: return int(encoders[col].transform([val])[0])
            except: return 0
        inp = {'Age':age,'Gender':enc('Gender',gender),'BMI':bmi,'Sleep_Hours_Per_Day':sleep,
               'Blood_Pressure':bp,'Cholesterol_Level':chol,'Triglyceride_Level':trig,
               'Fasting_Blood_Sugar':fbs,'CRP_Level':crp,'Homocysteine_Level':homo,
               'Smoking':enc('Smoking',smoking),'Diabetes':enc('Diabetes',diabetes),
               'Family_History':enc('Family_History',fam_hist),'High_Blood_Pressure':enc('High_Blood_Pressure',high_bp),
               'Low_HDL_Cholesterol':enc('Low_HDL_Cholesterol',low_hdl),'High_LDL_Cholesterol':enc('High_LDL_Cholesterol',high_ldl),
               'Exercise_Habits':enc('Exercise_Habits',exercise),'Alcohol_Consumption':enc('Alcohol_Consumption',alcohol),
               'Stress_Level':enc('Stress_Level',stress),'Sugar_Consumption':enc('Sugar_Consumption',sugar)}
        df_in = pd.DataFrame([inp])[feature_cols]
        scaled = scaler.transform(df_in)
        prob = float(model.predict_proba(scaled)[0][1])
        pred_val = int(model.predict(scaled)[0])
        pct = round(prob*100,1)
        if pred_val==1:
            level = "🔴 HIGH RISK" if pct>=70 else "⚡ MEDIUM-HIGH RISK"
            st.markdown(f'<div class="result-high"><span class="r-emoji">🔥</span><p class="r-verdict">HIGH RISK DETECTED</p><p class="r-prob">{pct}%</p><p class="r-caption">Probability of developing heart disease</p><div class="r-bar-wrap"><div class="r-bar-labels"><span>Risk Score</span><span>{pct}%</span></div><div class="r-bar-track"><div class="r-bar-fill" style="width:{pct}%"></div></div></div><span class="r-pill">{level}</span></div>', unsafe_allow_html=True)
        else:
            level = "✔ LOW RISK" if pct<30 else "⚡ LOW-MEDIUM RISK"
            st.markdown(f'<div class="result-low"><span class="r-emoji">✅</span><p class="r-verdict">LOW RISK</p><p class="r-prob">{pct}%</p><p class="r-caption">Probability of developing heart disease</p><div class="r-bar-wrap"><div class="r-bar-labels"><span>Risk Score</span><span>{pct}%</span></div><div class="r-bar-track"><div class="r-bar-fill" style="width:{pct}%"></div></div></div><span class="r-pill">{level}</span></div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.74rem;color:#9ca3af;text-align:center;margin-top:10px;">⚕️ This tool is for educational purposes only. Always consult a qualified medical professional.</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="app-footer">❤️ Heart Disease Prediction &nbsp;·&nbsp; Powered by XGBoost Machine Learning &nbsp;·&nbsp; For educational use only</div>', unsafe_allow_html=True)
