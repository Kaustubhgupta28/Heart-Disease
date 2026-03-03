import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

@st.cache_resource
def train_model():
    df = pd.read_csv("heart_disease.csv")

    # Missing values
    df['Alcohol Consumption'] = df['Alcohol Consumption'].fillna('Unknown')
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Target
    df['Heart Disease Status'] = (df['Heart Disease Status'] == 'Yes').astype(int)

    # Binary encoding
    df['Gender'] = (df['Gender'] == 'Male').astype(int)
    yes_no_cols = ['Smoking','Family Heart Disease','Diabetes',
                   'High Blood Pressure','Low HDL Cholesterol','High LDL Cholesterol']
    for col in yes_no_cols:
        df[col] = (df[col] == 'Yes').astype(int)

    # Ordinal encoding
    ord_map = {'Low': 0, 'Medium': 1, 'High': 2}
    for col in ['Exercise Habits', 'Stress Level', 'Sugar Consumption']:
        df[col] = df[col].map(ord_map)
    df['Alcohol Consumption'] = df['Alcohol Consumption'].map({'Unknown': -1, 'Low': 0, 'Medium': 1, 'High': 2})

    # Feature engineering
    df['BMI_BP_ratio']         = df['BMI'] / (df['Blood Pressure'] + 1)
    df['Lipid_risk_score']     = df['Cholesterol Level'] + df['Triglyceride Level']
    df['Metabolic_risk_score'] = df['Fasting Blood Sugar'] + df['CRP Level'] + df['Homocysteine Level']

    # Outlier capping
    clip_cols = ['Blood Pressure','Cholesterol Level','BMI','Triglyceride Level',
                 'Fasting Blood Sugar','CRP Level','Homocysteine Level','Sleep Hours','Age']
    clip_bounds = {}
    for col in clip_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        clip_bounds[col] = (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        df[col] = df[col].clip(*clip_bounds[col])

    # Scaling
    scale_cols = ['Age','Blood Pressure','Cholesterol Level','BMI','Sleep Hours',
                  'Triglyceride Level','Fasting Blood Sugar','CRP Level','Homocysteine Level',
                  'BMI_BP_ratio','Lipid_risk_score','Metabolic_risk_score']
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    feature_columns = [c for c in df.columns if c != 'Heart Disease Status']
    X = df[feature_columns]
    y = df['Heart Disease Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                          scale_pos_weight=4.0, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)

    return model, scaler, clip_bounds, feature_columns

# ── Train on startup ──────────────────────────────────────────────
with st.spinner("⏳ Loading model... please wait"):
    model, scaler, clip_bounds, feature_columns = train_model()

# ── UI ───────────────────────────────────────────────────────────
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("Fill in your details below and click **Predict** to see your heart disease risk.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age        = st.number_input("Age",                min_value=18, max_value=100, value=45)
    gender     = st.selectbox("Gender",               ["Male", "Female"])
    bp         = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=250, value=130)
    cholesterol= st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=400, value=200)
    bmi        = st.number_input("BMI",               min_value=10.0, max_value=60.0, value=25.0)
    sleep      = st.number_input("Sleep Hours",       min_value=3.0,  max_value=12.0, value=7.0)
    triglyceride = st.number_input("Triglyceride Level (mg/dL)", min_value=50, max_value=600, value=150)
    fbs        = st.number_input("Fasting Blood Sugar (mg/dL)",  min_value=50, max_value=300, value=90)
    crp        = st.number_input("CRP Level (mg/L)",  min_value=0.0,  max_value=20.0, value=1.0)
    homo       = st.number_input("Homocysteine Level (μmol/L)", min_value=3.0, max_value=30.0, value=8.0)

with col2:
    exercise   = st.selectbox("Exercise Habits",      ["Low", "Medium", "High"])
    smoking    = st.selectbox("Smoking",              ["No", "Yes"])
    family_hd  = st.selectbox("Family Heart Disease", ["No", "Yes"])
    diabetes   = st.selectbox("Diabetes",             ["No", "Yes"])
    high_bp    = st.selectbox("High Blood Pressure",  ["No", "Yes"])
    low_hdl    = st.selectbox("Low HDL Cholesterol",  ["No", "Yes"])
    high_ldl   = st.selectbox("High LDL Cholesterol", ["No", "Yes"])
    alcohol    = st.selectbox("Alcohol Consumption",  ["Unknown", "Low", "Medium", "High"])
    stress     = st.selectbox("Stress Level",         ["Low", "Medium", "High"])
    sugar      = st.selectbox("Sugar Consumption",    ["Low", "Medium", "High"])

st.markdown("---")

if st.button("🔍 Predict My Heart Disease Risk", use_container_width=True):

    # Build raw input
    mp = pd.DataFrame([{
        'Age': age, 'Gender': gender, 'Blood Pressure': bp,
        'Cholesterol Level': cholesterol, 'Exercise Habits': exercise,
        'Smoking': smoking, 'Family Heart Disease': family_hd,
        'Diabetes': diabetes, 'BMI': bmi, 'High Blood Pressure': high_bp,
        'Low HDL Cholesterol': low_hdl, 'High LDL Cholesterol': high_ldl,
        'Alcohol Consumption': alcohol, 'Stress Level': stress,
        'Sleep Hours': sleep, 'Sugar Consumption': sugar,
        'Triglyceride Level': triglyceride, 'Fasting Blood Sugar': fbs,
        'CRP Level': crp, 'Homocysteine Level': homo
    }])

    # Encode
    mp['Gender'] = (mp['Gender'] == 'Male').astype(int)
    for col in ['Smoking','Family Heart Disease','Diabetes',
                'High Blood Pressure','Low HDL Cholesterol','High LDL Cholesterol']:
        mp[col] = (mp[col] == 'Yes').astype(int)

    ord_map = {'Low': 0, 'Medium': 1, 'High': 2}
    for col in ['Exercise Habits', 'Stress Level', 'Sugar Consumption']:
        mp[col] = mp[col].map(ord_map)
    mp['Alcohol Consumption'] = mp['Alcohol Consumption'].map({'Unknown': -1, 'Low': 0, 'Medium': 1, 'High': 2})

    # Feature engineering
    mp['BMI_BP_ratio']         = mp['BMI'] / (mp['Blood Pressure'] + 1)
    mp['Lipid_risk_score']     = mp['Cholesterol Level'] + mp['Triglyceride Level']
    mp['Metabolic_risk_score'] = mp['Fasting Blood Sugar'] + mp['CRP Level'] + mp['Homocysteine Level']

    # Clip + scale
    clip_cols = ['Blood Pressure','Cholesterol Level','BMI','Triglyceride Level',
                 'Fasting Blood Sugar','CRP Level','Homocysteine Level','Sleep Hours','Age']
    for col in clip_cols:
        mp[col] = mp[col].clip(*clip_bounds[col])

    scale_cols = ['Age','Blood Pressure','Cholesterol Level','BMI','Sleep Hours',
                  'Triglyceride Level','Fasting Blood Sugar','CRP Level','Homocysteine Level',
                  'BMI_BP_ratio','Lipid_risk_score','Metabolic_risk_score']
    mp[scale_cols] = scaler.transform(mp[scale_cols])

    # Predict
    prob = model.predict_proba(mp[feature_columns])[0][1]
    pred = model.predict(mp[feature_columns])[0]

    # Result
    st.markdown("---")
    if pred == 1:
        st.error(f"### ❤️ High Risk of Heart Disease")
        st.markdown(f"**Disease Probability: {prob*100:.1f}%**")
    else:
        st.success(f"### ✅ Low Risk of Heart Disease")
        st.markdown(f"**Disease Probability: {prob*100:.1f}%**")

    # Risk bar
    risk_color = "🔴" if prob > 0.6 else "🟡" if prob > 0.3 else "🟢"
    risk_label = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
    st.markdown(f"**Risk Level: {risk_color} {risk_label}**")
    st.progress(float(prob))

st.markdown("---")
st.caption("⚠️ This tool is for educational purposes only. Always consult a medical professional.")
