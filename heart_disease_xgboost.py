import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@700;800;900&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
[data-testid="stToolbar"] {display: none;}
.stApp > header {display: none;}
.stApp {background-color: #f0f2f5 !important;}
.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}
.hero {
    background: linear-gradient(135deg, #7b0020, #9b0030, #6d001c) !important;
    padding: 60px 80px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    min-height: 300px !important;
    position: relative !important;
    overflow: hidden !important;
}
.hero h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 3.5rem !important;
    font-weight: 900 !important;
    color: white !important;
    line-height: 1.1 !important;
    margin: 0 0 16px 0 !important;
}
.hero p {
    font-family: 'Inter', sans-serif !important;
    color: rgba(255,255,255,0.6) !important;
    font-size: 1rem !important;
    font-weight: 300 !important;
    line-height: 1.7 !important;
    margin: 0 !important;
}
.hero-heart { font-size: 9rem !important; animation: hb 2s ease-in-out infinite !important; }
@keyframes hb { 0%,100%{transform:scale(1)} 50%{transform:scale(1.08)} }
.vdot { position: absolute !important; border-radius: 50% !important; background: radial-gradient(circle at 30% 30%, rgba(255,150,150,0.5), rgba(180,30,60,0.7)) !important; }
.vd1{width:55px;height:55px;top:15%;right:8%;}
.vd2{width:38px;height:38px;bottom:15%;right:4%;}
.vd3{width:25px;height:25px;top:20%;right:28%;}
.wrap {padding: 40px 80px 60px !important;}
.sec-head { display: flex !important; align-items: center !important; gap: 14px !important; margin: 32px 0 16px !important; }
.sec-icon { width: 40px !important; height: 40px !important; border-radius: 10px !important; display: flex !important; align-items: center !important; justify-content: center !important; font-size: 1.1rem !important; flex-shrink: 0 !important; }
.ic-red   {background: linear-gradient(135deg,#9b0030,#c1121f) !important;}
.ic-blue  {background: linear-gradient(135deg,#1d4ed8,#3b82f6) !important;}
.ic-green {background: linear-gradient(135deg,#15803d,#22c55e) !important;}
.ic-orange{background: linear-gradient(135deg,#c2410c,#f97316) !important;}
.ic-purple{background: linear-gradient(135deg,#7c3aed,#a78bfa) !important;}
.sec-title {font-family:'Outfit',sans-serif !important; font-size:1.1rem !important; font-weight:700 !important; color:#111 !important; margin:0 !important;}
.sec-sub   {font-size:0.75rem !important; color:#888 !important; margin:2px 0 0 !important;}
.card { background: white !important; border-radius: 16px !important; padding: 28px !important; border: 1px solid #e5e7eb !important; box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important; margin-bottom: 8px !important; }
div[data-testid="stNumberInput"] input { background: #f9fafb !important; border: 1.5px solid #e5e7eb !important; border-radius: 10px !important; color: #111 !important; }
div[data-testid="stSelectbox"] > div > div { background: #f9fafb !important; border: 1.5px solid #e5e7eb !important; border-radius: 10px !important; }
div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label { font-size: 0.75rem !important; font-weight: 600 !important; color: #666 !important; text-transform: uppercase !important; letter-spacing: 0.04em !important; }
div[data-testid="stButton"] > button { background: linear-gradient(135deg, #9b0030, #c1121f) !important; color: white !important; border: none !important; border-radius: 14px !important; padding: 16px 40px !important; font-family: 'Outfit', sans-serif !important; font-size: 1rem !important; font-weight: 700 !important; letter-spacing: 0.06em !important; width: 100% !important; box-shadow: 0 4px 20px rgba(155,0,48,0.4) !important; }
div[data-testid="stButton"] > button:hover { transform: translateY(-2px) !important; }
.res-high { background: linear-gradient(135deg,#7b0020,#c1121f) !important; border-radius: 20px !important; padding: 40px !important; text-align: center !important; color: white !important; box-shadow: 0 12px 40px rgba(155,0,48,0.4) !important; margin: 20px 0 !important; }
.res-low  { background: linear-gradient(135deg,#14532d,#15803d) !important; border-radius: 20px !important; padding: 40px !important; text-align: center !important; color: white !important; box-shadow: 0 12px 40px rgba(21,128,61,0.38) !important; margin: 20px 0 !important; }
.r-emoji {font-size:3rem; display:block; margin-bottom:10px;}
.r-title {font-family:'Outfit',sans-serif; font-size:2rem; font-weight:900; margin:0 0 6px;}
.r-pct   {font-family:'Outfit',sans-serif; font-size:5rem; font-weight:900; line-height:1; margin:10px 0;}
.r-cap   {font-size:0.85rem; opacity:0.6;}
.r-track {height:10px; background:rgba(255,255,255,0.15); border-radius:999px; overflow:hidden; margin:18px auto; max-width:360px;}
.r-fill  {height:100%; border-radius:999px; background:linear-gradient(90deg,rgba(255,255,255,0.4),white);}
.r-pill  {display:inline-block; border-radius:999px; padding:6px 20px; font-weight:700; font-size:0.78rem; letter-spacing:0.1em; margin-top:14px; background:rgba(255,255,255,0.15); color:white; border:1px solid rgba(255,255,255,0.3);}

/* CAUSES GRID */
.causes-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-top: 4px; }
.cause-card { background: white; border-radius: 14px; padding: 20px; border: 1px solid #e5e7eb; box-shadow: 0 2px 8px rgba(0,0,0,0.05); transition: transform 0.2s; }
.cause-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.1); }
.cause-icon { font-size: 2rem; margin-bottom: 10px; display: block; }
.cause-title { font-family:'Outfit',sans-serif; font-size:0.95rem; font-weight:700; color:#1a1a2e; margin-bottom:6px; }
.cause-desc  { font-size:0.78rem; color:#6b7280; line-height:1.6; }

/* SUGGESTIONS */
.sug-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 14px; margin-top: 4px; }
.sug-card { border-radius: 14px; padding: 18px 20px; border-left: 4px solid; }
.sug-card.red    { background:#fff5f5; border-color:#e63946; }
.sug-card.orange { background:#fff7ed; border-color:#f97316; }
.sug-card.green  { background:#f0fdf4; border-color:#22c55e; }
.sug-card.blue   { background:#eff6ff; border-color:#3b82f6; }
.sug-card.purple { background:#f5f3ff; border-color:#8b5cf6; }
.sug-card.teal   { background:#f0fdfa; border-color:#14b8a6; }
.sug-title { font-family:'Outfit',sans-serif; font-size:0.9rem; font-weight:700; margin-bottom:5px; color:#1a1a2e; display:flex; align-items:center; gap:8px; }
.sug-text  { font-size:0.78rem; color:#4b5563; line-height:1.6; }

.warn {background:#fffbeb; border-left:4px solid #f59e0b; border-radius:10px; padding:14px 18px; font-size:0.85rem; color:#92400e; margin:12px 0;}
.foot {background:#1a1a2e; color:rgba(255,255,255,0.3); text-align:center; padding:24px; font-size:0.76rem; margin-top:60px; letter-spacing:0.03em;}
</style>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="vdot vd1"></div><div class="vdot vd2"></div><div class="vdot vd3"></div>
  <div>
    <h1>Heart Disease<br>Prediction</h1>
    <p>Analyzing risk factors to forecast likelihood of<br>developing heart conditions.</p>
  </div>
  <div class="hero-heart">🫀</div>
</div>
""", unsafe_allow_html=True)

# ── MODEL ─────────────────────────────────────────────────────────────────────
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
        le = LabelEncoder(); df[c] = le.fit_transform(df[c]); encoders[c] = le
    X = df.drop('Heart_Disease',axis=1); y = df['Heart_Disease']
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    Xtr,_,ytr,_ = train_test_split(Xs,y,test_size=0.2,random_state=42)
    m = xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,random_state=42,use_label_encoder=False,eval_metric='logloss')
    m.fit(Xtr,ytr)
    return m, sc, encoders, X.columns.tolist()

model, scaler, encoders, feature_cols = load_model()

st.markdown('<div class="wrap">', unsafe_allow_html=True)

# ── CAUSES OF HEART DISEASE ───────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
  <div class="sec-icon ic-orange">⚠️</div>
  <div><p class="sec-title">Common Causes of Heart Disease</p>
  <p class="sec-sub">Understanding the key risk factors that contribute to heart conditions</p></div>
</div>
<div class="card">
<div class="causes-grid">
  <div class="cause-card">
    <span class="cause-icon">🚬</span>
    <p class="cause-title">Smoking</p>
    <p class="cause-desc">Tobacco damages blood vessel walls, reduces oxygen in blood, and raises blood pressure — significantly increasing heart attack risk.</p>
  </div>
  <div class="cause-card">
    <span class="cause-icon">🩸</span>
    <p class="cause-title">High Blood Pressure</p>
    <p class="cause-desc">Hypertension forces the heart to work harder, thickening the heart muscle and hardening arteries over time.</p>
  </div>
  <div class="cause-card">
    <span class="cause-icon">🍔</span>
    <p class="cause-title">High Cholesterol</p>
    <p class="cause-desc">Excess LDL cholesterol builds up as plaque inside arteries, narrowing them and restricting blood flow to the heart.</p>
  </div>
  <div class="cause-card">
    <span class="cause-icon">🍩</span>
    <p class="cause-title">Diabetes</p>
    <p class="cause-desc">High blood sugar damages blood vessels and nerves controlling the heart, doubling the risk of cardiovascular disease.</p>
  </div>
  <div class="cause-card">
    <span class="cause-icon">😰</span>
    <p class="cause-title">Chronic Stress</p>
    <p class="cause-desc">Ongoing stress raises cortisol and adrenaline, increasing blood pressure and promoting inflammation in arteries.</p>
  </div>
  <div class="cause-card">
    <span class="cause-icon">🛋️</span>
    <p class="cause-title">Physical Inactivity</p>
    <p class="cause-desc">A sedentary lifestyle leads to obesity, high blood pressure, and poor cholesterol levels — all major heart risk factors.</p>
  </div>
  <div class="cause-card">
    <span class="cause-icon">🧬</span>
    <p class="cause-title">Family History</p>
    <p class="cause-desc">Genetics play a major role. If a close relative had heart disease, your risk is significantly higher.</p>
  </div>
  <div class="cause-card">
    <span class="cause-icon">⚖️</span>
    <p class="cause-title">Obesity</p>
    <p class="cause-desc">Excess body weight strains the heart, raises blood pressure, and is linked to diabetes and high cholesterol.</p>
  </div>
  <div class="cause-card">
    <span class="cause-icon">🍺</span>
    <p class="cause-title">Excess Alcohol</p>
    <p class="cause-desc">Heavy drinking weakens the heart muscle, raises blood pressure, and contributes to irregular heartbeat.</p>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

# ── FORM ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
  <div class="sec-icon ic-red">👤</div>
  <div><p class="sec-title">Personal Information</p><p class="sec-sub">Basic demographic and physical details</p></div>
</div><div class="card">""", unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
with c1: age = st.number_input("AGE (YEARS)", min_value=0, max_value=120, value=0)
with c2: gender = st.selectbox("GENDER", ["Select","Male","Female"])
with c3: bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0, step=0.1)
with c4: sleep = st.number_input("SLEEP HOURS/DAY", min_value=0, max_value=24, value=0)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="sec-head">
  <div class="sec-icon ic-blue">🔬</div>
  <div><p class="sec-title">Clinical Measurements</p><p class="sec-sub">Lab results and vital signs</p></div>
</div><div class="card">""", unsafe_allow_html=True)
c1,c2,c3 = st.columns(3)
with c1:
    bp = st.number_input("BLOOD PRESSURE (mmHg)", min_value=0, max_value=300, value=0)
    fbs = st.number_input("FASTING BLOOD SUGAR (mg/dL)", min_value=0, max_value=500, value=0)
with c2:
    chol = st.number_input("CHOLESTEROL (mg/dL)", min_value=0, max_value=600, value=0)
    crp = st.number_input("CRP LEVEL (mg/L)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
with c3:
    trig = st.number_input("TRIGLYCERIDE (mg/dL)", min_value=0, max_value=1000, value=0)
    homo = st.number_input("HOMOCYSTEINE (µmol/L)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="sec-head">
  <div class="sec-icon ic-green">🏃</div>
  <div><p class="sec-title">Health &amp; Lifestyle</p><p class="sec-sub">Medical history and daily habits</p></div>
</div><div class="card">""", unsafe_allow_html=True)
c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    smoking = st.selectbox("SMOKING", ["Select","Yes","No"])
    diabetes = st.selectbox("DIABETES", ["Select","Yes","No"])
with c2:
    fam_hist = st.selectbox("FAMILY HISTORY", ["Select","Yes","No"])
    high_bp = st.selectbox("HIGH BP", ["Select","Yes","No"])
with c3:
    low_hdl = st.selectbox("LOW HDL", ["Select","Yes","No"])
    high_ldl = st.selectbox("HIGH LDL", ["Select","Yes","No"])
with c4:
    exercise = st.selectbox("EXERCISE", ["Select","Regular","Occasional","None"])
    alcohol = st.selectbox("ALCOHOL", ["Select","Heavy","Moderate","None"])
with c5:
    stress = st.selectbox("STRESS LEVEL", ["Select","High","Medium","Low"])
    sugar = st.selectbox("SUGAR INTAKE", ["Select","High","Medium","Low"])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
_,btn_col,_ = st.columns([1,2,1])
with btn_col:
    predict = st.button("❤️  ANALYZE HEART DISEASE RISK")

# ── PREDICTION ───────────────────────────────────────────────────────────────
if predict:
    selects = [gender,smoking,diabetes,fam_hist,high_bp,low_hdl,high_ldl,exercise,alcohol,stress,sugar]
    nums = [age,bmi,bp,chol,trig,fbs,crp,homo,sleep]
    if any(s=="Select" for s in selects) or any(v==0 for v in nums):
        st.markdown('<div class="warn">⚠️ <strong>Incomplete form.</strong> Please fill in all fields before predicting.</div>', unsafe_allow_html=True)
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

        # ── RESULT CARD ──
        if pred_val==1:
            level = "🔴 HIGH RISK" if pct>=70 else "⚡ MEDIUM-HIGH RISK"
            st.markdown(f"""
            <div class="res-high">
              <span class="r-emoji">🔥</span>
              <p class="r-title">HIGH RISK DETECTED</p>
              <p class="r-pct">{pct}%</p>
              <p class="r-cap">Probability of developing heart disease</p>
              <div class="r-track"><div class="r-fill" style="width:{pct}%"></div></div>
              <span class="r-pill">{level}</span>
            </div>""", unsafe_allow_html=True)
        else:
            level = "✔ LOW RISK" if pct<30 else "⚡ LOW-MEDIUM RISK"
            st.markdown(f"""
            <div class="res-low">
              <span class="r-emoji">✅</span>
              <p class="r-title">LOW RISK</p>
              <p class="r-pct">{pct}%</p>
              <p class="r-cap">Probability of developing heart disease</p>
              <div class="r-track"><div class="r-fill" style="width:{pct}%"></div></div>
              <span class="r-pill">{level}</span>
            </div>""", unsafe_allow_html=True)

        # ── PERSONALISED SUGGESTIONS ──
        suggestions = []

        if smoking == "Yes":
            suggestions.append(("red","🚬","Quit Smoking Immediately","Smoking is the #1 modifiable risk factor. Quitting within 1 year halves your heart disease risk. Seek nicotine replacement therapy or consult your doctor."))
        if bp > 140 or high_bp == "Yes":
            suggestions.append(("orange","🩸","Lower Your Blood Pressure","Your BP is elevated. Reduce salt intake to under 5g/day, avoid caffeine, exercise regularly, and take prescribed medication consistently."))
        if chol > 240 or high_ldl == "Yes":
            suggestions.append(("orange","🥑","Improve Your Cholesterol","Cut saturated fats, eat more oats, nuts and olive oil. Omega-3 fish like salmon help raise good HDL. Ask your doctor about statins if needed."))
        if bmi > 30:
            suggestions.append(("red","⚖️","Manage Your Weight","A BMI above 30 significantly strains your heart. Aim to lose 5–10% of body weight through a calorie-controlled diet and daily activity."))
        if exercise == "None":
            suggestions.append(("blue","🏃","Start Exercising Regularly","Aim for at least 150 minutes of moderate activity per week (brisk walking, cycling, swimming). Even 30 mins/day reduces heart risk by 35%."))
        if exercise == "Occasional":
            suggestions.append(("blue","🏃","Increase Exercise Frequency","Move from occasional to regular exercise. Try to be active 5 days a week. Even short 10-minute walks add up to big heart benefits."))
        if stress == "High":
            suggestions.append(("purple","🧘","Manage Stress Actively","Chronic stress raises cortisol and damages arteries. Practice daily meditation, deep breathing, yoga, or spend time in nature to lower stress hormones."))
        if diabetes == "Yes" or fbs > 126:
            suggestions.append(("teal","💉","Control Blood Sugar","High blood sugar damages blood vessels. Follow a low-glycaemic diet, monitor glucose regularly, take medication as prescribed, and exercise daily."))
        if alcohol == "Heavy":
            suggestions.append(("orange","🍺","Reduce Alcohol Intake","Heavy drinking weakens the heart muscle. Limit to max 1 drink/day for women and 2 for men, or ideally avoid completely."))
        if sleep < 6:
            suggestions.append(("blue","😴","Improve Sleep Quality","Less than 6 hours of sleep raises heart disease risk by 48%. Establish a consistent sleep schedule and avoid screens 1 hour before bed."))
        if low_hdl == "Yes":
            suggestions.append(("green","🥜","Boost Your HDL Cholesterol","Low HDL means less protection for your heart. Eat more healthy fats (avocado, nuts, olive oil), exercise regularly, and quit smoking."))
        if trig > 200:
            suggestions.append(("orange","🐟","Lower Triglycerides","Cut sugar, refined carbs and alcohol. Eat fatty fish twice a week, take omega-3 supplements, and increase physical activity."))
        if sugar == "High":
            suggestions.append(("red","🍬","Cut Down on Sugar","High sugar consumption leads to obesity, diabetes and inflammation — all major heart risk factors. Replace sweets with fruits and whole grains."))

        # Always add positive reinforcement
        if pred_val == 0:
            suggestions.append(("green","✅","Keep Up the Good Work!","Your results look positive! Maintain your healthy habits — regular exercise, balanced diet and routine check-ups will keep your heart strong."))
            suggestions.append(("teal","🏥","Schedule Regular Check-ups","Even with low risk, get a heart health check every 1–2 years. Early detection is the best protection."))

        if suggestions:
            st.markdown("""
            <div class="sec-head">
              <div class="sec-icon ic-purple">💡</div>
              <div><p class="sec-title">Personalised Health Suggestions</p>
              <p class="sec-sub">Based on your results — steps you can take to improve your heart health</p></div>
            </div>
            <div class="card">
            <div class="sug-grid">
            """, unsafe_allow_html=True)
            for color, icon, title, text in suggestions:
                st.markdown(f"""
                <div class="sug-card {color}">
                  <p class="sug-title">{icon} {title}</p>
                  <p class="sug-text">{text}</p>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

        st.markdown('<p style="font-size:0.74rem;color:#9ca3af;text-align:center;margin-top:16px;">⚕️ For educational use only. Always consult a qualified medical professional for diagnosis and treatment.</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="foot">❤️ Heart Disease Prediction &nbsp;·&nbsp; Powered by XGBoost &nbsp;·&nbsp; For educational use only</div>', unsafe_allow_html=True)
