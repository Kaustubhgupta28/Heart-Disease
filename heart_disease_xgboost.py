import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide", initial_sidebar_state="collapsed")

# Static CSS styling
CSS = """
<style>
/* Add your styles here */
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ── Introduction Animation ──────────
components.html("""
<!DOCTYPE html>
<html>
<head>
<style>
/* Add animation styles here */
</style>
</head>
<body>
<script>
// Add your animation JavaScript here
</script>
</body>
</html>
""", height=1, scrolling=False)

# ── Load Model Function ──────────
@st.cache_resource
def load_model():
    # Loading and preparing the model as per your previous implementation
    np.random.seed(42)
    n = 8000
    data = {
        'Age': np.random.randint(20,80,n),
        'Gender': np.random.choice(['Male','Female'],n),
        # Add more features if necessary
    }
    df = pd.DataFrame(data)
    # Dummy risk generation logic
    df['Heart_Disease'] = np.random.choice([0, 1], n)
    
    # Encoding and splitting data
    cat_cols = ['Gender']
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder(); df[c] = le.fit_transform(df[c]); encoders[c] = le
        
    X = df.drop('Heart_Disease', axis=1)
    y = df['Heart_Disease']
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(Xs, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model, sc, encoders, X.columns.tolist()

model, scaler, encoders, feature_cols = load_model()

# ── User Input ──────────

st.markdown('<div class="wrap">', unsafe_allow_html=True)
# Add your input fields and forms based on previous implementation...
age = st.number_input("Age", 0, 120)
gender = st.selectbox("Gender", ["Select", "Male", "Female"])
# Add other input fields...

# ── Prediction ──────────
if st.button("Analyze Heart Disease Risk"):
    # Add your prediction logic...
    # Display prediction results...

# ── Visualization ──────────
if st.button("Show Risk Factor Comparison"):
    avg_risks = {
        'Age': 30,
        'Gender': 'Male',
        'Cholesterol': 200,
        # other average risks
    }
    
    user_input = {
        'Age': age,
        'Gender': gender,
        # Capture other user inputs
    }
    # Plotting
    plt.bar(user_input.keys(), [user_input[key] if key != 'Gender' else 1 for key in user_input.keys()], alpha=0.5)
    plt.axhline(y=avg_risks['Age'], label='Average Risk (Age)', color='r', linestyle='--')
    plt.title('User Input vs Average Risk Factors')
    st.pyplot(plt)

# ── User Feedback ──────────
st.subheader("We value your feedback!")
feedback = st.text_area("Please share your comments or suggestions:")
if st.button("Submit Feedback"):
    # Here you would save feedback to a file or database
    st.success("Thank you for your feedback!")

st.markdown('</div>', unsafe_allow_html=True)
