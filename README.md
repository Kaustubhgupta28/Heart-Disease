# 🫀 Heart Disease Risk Prediction

A Machine Learning web application that predicts the risk of heart disease based on 15+ patient health parameters including clinical measurements and lifestyle factors.

---

## 📌 Overview

Heart disease is one of the leading causes of death worldwide. This project leverages Machine Learning to assist in **early detection** of heart disease risk using patient data such as blood pressure, cholesterol, BMI, and lifestyle habits — providing real-time predictions through a Streamlit-based web interface.

---

## 🚀 Features

- Predicts heart disease risk based on 15+ health parameters
- Real-time predictions via interactive web interface
- Handles both clinical data (BP, cholesterol, CRP) and lifestyle factors (smoking, exercise, stress)
- Data preprocessing with feature scaling and class balancing (SMOTE)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python |
| ML Library | Scikit-learn |
| Model | XGBoost |
| Web Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## 📊 Dataset

- **Source:** Kaggle (`heart_disease`)
- **Size:** 10,000 patient records
- **Features:** 15+ parameters including:
  - Blood Pressure, Cholesterol, Triglyceride
  - Fasting Blood Sugar, CRP Level, Homocysteine
  - BMI
  - Smoking, Diabetes, Family History
  - Exercise, Alcohol, Stress Level, Sugar Intake

---

## 🧠 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 81% |
| Algorithm | XGBoost |

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/Kaustubhgupta28/Heart-Disease.git
cd Heart-Disease

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run heart_disease_xgboost.py
```

Then open your browser and go to: `http://localhost:8501`

---

## 📁 Project Structure

```
Heart-Disease/
│
├── heart_disease_xgboost.py  # Streamlit web application + XGBoost model
├── heart_disease.csv         # Kaggle dataset
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 📷 Demo

> Add screenshot of your web interface here

---

## 👤 Author

**Kaustubh Gupta**
- 🔗 [LinkedIn](https://linkedin.com/in/kaustubh-gupta-498b7a273)
- 🐙 [GitHub](https://github.com/Kaustubhgupta28)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
