#!/usr/bin/env python
# coding: utf-8

# In[85]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier, plot_importance


# In[86]:


df = pd.read_csv('heart_disease.csv')

print(f'Shape: {df.shape}')
print(f'Rows: {df.shape[0]:,} | Columns: {df.shape[1]}')
df.head()


# ## Exploratory Data Analysis (EDA)

# In[87]:


print(df.dtypes)
print()

df.describe(include='all').T


# In[88]:


# Missing value heatmap
missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df  = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})

print(missing_df[missing_df['Missing Count'] > 0])

plt.figure(figsize=(12, 5))
missing_pct[missing_pct > 0].sort_values(ascending=False).plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Missing Value Percentage by Column', fontsize=14)
plt.ylabel('Missing %')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(plt)


# In[89]:


# Target distribution
plt.figure(figsize=(6, 4))
df['Heart Disease Status'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'], edgecolor='black')
plt.title('Target Variable Distribution', fontsize=14)
plt.xlabel('Heart Disease Status')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print(df['Heart Disease Status'].value_counts())


# ## Handle Missing Values

# In[90]:


df['Alcohol Consumption'] = df['Alcohol Consumption'].fillna('Unknown')


# In[91]:


# Numerical columns → median imputation

num_cols = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI',
            'Sleep Hours', 'Triglyceride Level', 'Fasting Blood Sugar',
            'CRP Level', 'Homocysteine Level']
for col in num_cols:
    n = df[col].isnull().sum()
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)


# In[92]:


# Binary Yes/No categorical → mode imputation
bin_cat_cols = ['Gender', 'Smoking', 'Family Heart Disease', 'Diabetes',
                'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol']
for col in bin_cat_cols:
    n = df[col].isnull().sum()
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)


# In[93]:


# Ordinal categorical → mode imputation
for col in ['Exercise Habits', 'Stress Level', 'Sugar Consumption']:
    n = df[col].isnull().sum()
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)


# In[94]:


df.isnull().sum()


# ## Encode Target Variable

# In[95]:


df['Heart Disease Status'] = (df['Heart Disease Status'] == 'Yes').astype(int)

print('Target encoded: No → 0, Yes → 1')
print(df['Heart Disease Status'].value_counts())


# ## Binary Encoding (Yes/No Columns)

# In[96]:


# Gender: Male=1, Female=0
df['Gender'] = (df['Gender'] == 'Male').astype(int)
print('Gender → (Male=1, Female=0)')


# In[97]:


# Yes/No columns → 1/0
yes_no_cols = ['Smoking', 'Family Heart Disease', 'Diabetes',
               'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol']
for col in yes_no_cols:
    df[col] = (df[col] == 'Yes').astype(int)
    print(f'{col} → Yes=1, No=0')


# In[98]:


df.head()


# ## Ordinal Encoding

# In[99]:


# Low=0, Medium=1, High=2
ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}


# In[100]:


for col in ['Exercise Habits', 'Stress Level', 'Sugar Consumption']:
    df[col] = df[col].map(ordinal_map)
    print(f'{col} → Low=0, Medium=1, High=2')


# In[101]:


# Alcohol Consumption: Unknown=-1, Low=0, Medium=1, High=2
alcohol_map = {'Unknown': -1, 'Low': 0, 'Medium': 1, 'High': 2}
df['Alcohol Consumption'] = df['Alcohol Consumption'].map(alcohol_map)
print('Alcohol Consumption → Unknown=-1, Low=0, Medium=1, High=2')


# In[102]:


df.head()


# ##  Feature Engineering

# In[103]:


# BMI to Blood Pressure ratio — combined cardiovascular load
df['BMI_BP_ratio'] = df['BMI'] / (df['Blood Pressure'] + 1)

# Lipid risk score — total lipid burden
df['Lipid_risk_score'] = df['Cholesterol Level'] + df['Triglyceride Level']

# Metabolic risk score — inflammatory + glucose markers
df['Metabolic_risk_score'] = df['Fasting Blood Sugar'] + df['CRP Level'] + df['Homocysteine Level']

print('3 new features created:')
print('  BMI_BP_ratio         = BMI / (Blood Pressure + 1)')
print('  Lipid_risk_score     = Cholesterol Level + Triglyceride Level')
print('  Metabolic_risk_score = Fasting Blood Sugar + CRP Level + Homocysteine Level')
print(f'\nNew shape: {df.shape}')
df[['BMI_BP_ratio', 'Lipid_risk_score', 'Metabolic_risk_score']].describe()


# ## Outlier Capping (IQR Method)

# In[104]:


clip_cols = ['Blood Pressure', 'Cholesterol Level', 'BMI', 'Triglyceride Level',
             'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level',
             'Lipid_risk_score', 'Metabolic_risk_score']

print(f'{"Column":<25} {"Lower Cap":>12} {"Upper Cap":>12} {"Clipped Rows":>14}')
print('-' * 65)

for col in clip_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower, upper)
    print(f'{col:<25} {lower:>12.2f} {upper:>12.2f} {n_outliers:>14}')

print('\n✅ Outlier capping complete')


# In[105]:


fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, col in zip(axes.flatten(), clip_cols):
    ax.boxplot(df[col].dropna(), patch_artist=True,
               boxprops=dict(facecolor='lightblue'))
    ax.set_title(col, fontsize=9)
plt.suptitle('Boxplots After Outlier Capping', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()


# ## Feature Scaling (StandardScaler)

# In[106]:


scale_cols = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours',
              'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level',
              'Homocysteine Level', 'BMI_BP_ratio', 'Lipid_risk_score', 'Metabolic_risk_score']

scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

print('StandardScaler applied (mean=0, std=1) to numerical features:')
for col in scale_cols:
    print(f'  {col:<25}  mean={df[col].mean():+.4f}  std={df[col].std():.4f}')

print(f'\n✅ Scaling complete | Final dataset shape: {df.shape}')


# ## Train / Test Split

# In[107]:


X = df.drop(columns=['Heart Disease Status'])
y = df['Heart Disease Status']


# In[108]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ## XGBoost Model Training

# In[109]:


# Handle class imbalance with scale_pos_weight
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f'scale_pos_weight = {scale_pos_weight:.2f} (handles 4:1 class imbalance)')

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50
)


# In[110]:


# Training curve
results = xgb_model.evals_result()
plt.figure(figsize=(10, 4))
plt.plot(results['validation_0']['logloss'], label='Train Loss', color='steelblue')
plt.plot(results['validation_1']['logloss'], label='Test Loss', color='tomato')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.title('XGBoost Learning Curve')
plt.legend()
plt.tight_layout()
plt.show()


# ##  Model Evaluation

# In[111]:


y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
roc  = roc_auc_score(y_test, y_proba)

print('=' * 45)
print(f'  Accuracy : {acc:.4f}  ({acc*100:.2f}%)')
print(f'  ROC-AUC  : {roc:.4f}')
print('=' * 45)
print()
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Heart Disease']))


# In[112]:


# Predict on training data
y_train_pred =xgb_model.predict(X_train)

# Training accuracy
train_acc = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_acc)


# In[113]:


# Predict on test data
y_test_pred = xgb_model.predict(X_test)

# Testing accuracy
test_acc = accuracy_score(y_test, y_test_pred)
print("Testing Accuracy:", test_acc)


# In[114]:


# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Heart Disease'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix', fontsize=13)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, color='tomato', lw=2, label=f'ROC AUC = {roc:.4f}')
axes[1].plot([0, 1], [0, 1], color='grey', linestyle='--')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontsize=13)
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.show()


# In[115]:


# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

print('5-Fold Cross-Validation ROC-AUC:')
for i, score in enumerate(cv_scores, 1):
    print(f'  Fold {i}: {score:.4f}')
print(f'\n  Mean : {cv_scores.mean():.4f}')
print(f'  Std  : {cv_scores.std():.4f}')


# ## Feature Importance

# In[116]:


fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Built-in XGBoost importance
plot_importance(xgb_model, ax=axes[0], max_num_features=15,
                importance_type='gain', title='Feature Importance (Gain)',
                color='steelblue')

# Manual bar chart sorted
feat_imp = pd.Series(xgb_model.feature_importances_, index=X.columns)
feat_imp.sort_values(ascending=True).tail(15).plot(
    kind='barh', ax=axes[1], color='tomato', edgecolor='black'
)
axes[1].set_title('Top 15 Features (Weight)', fontsize=12)
axes[1].set_xlabel('Importance Score')

plt.tight_layout()
plt.show()

print('\nTop 10 most important features:')
print(feat_imp.sort_values(ascending=False).head(10).to_string())


# In[117]:


# ============================================================
# Cell — Define Training Reference Variables
# (reads fresh from raw CSV — no dependency on df state)
# ============================================================

raw = pd.read_csv('heart_disease.csv')   # reload raw data

num_cols = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours',
            'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level']

# Medians from raw training data
train_medians = {col: raw[col].median() for col in num_cols}

# Modes from raw training data
train_modes = {col: raw[col].mode()[0] for col in
               ['Gender', 'Smoking', 'Family Heart Disease', 'Diabetes',
                'High Blood Pressure', 'Low HDL Cholesterol', 'High LDL Cholesterol',
                'Exercise Habits', 'Stress Level', 'Sugar Consumption']}

# IQR clip bounds — compute on raw data after feature engineering
raw_temp = raw.copy()
raw_temp['Lipid_risk_score']     = raw_temp['Cholesterol Level'] + raw_temp['Triglyceride Level']
raw_temp['Metabolic_risk_score'] = raw_temp['Fasting Blood Sugar'] + raw_temp['CRP Level'] + raw_temp['Homocysteine Level']

clip_cols = ['Blood Pressure', 'Cholesterol Level', 'BMI', 'Triglyceride Level',
             'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level',
             'Lipid_risk_score', 'Metabolic_risk_score']

clip_bounds = {}
for col in clip_cols:
    Q1  = raw_temp[col].quantile(0.25)
    Q3  = raw_temp[col].quantile(0.75)
    IQR = Q3 - Q1
    clip_bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

scale_cols = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours',
              'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level',
              'Homocysteine Level', 'BMI_BP_ratio', 'Lipid_risk_score', 'Metabolic_risk_score']

feature_columns = X_train.columns.tolist()

print("✅ All reference variables defined from raw CSV")
print(f"   train_medians  : {len(train_medians)} cols")
print(f"   train_modes    : {len(train_modes)} cols")
print(f"   clip_bounds    : {len(clip_bounds)} cols")
print(f"   feature_columns: {len(feature_columns)} cols")


# In[130]:


multiple_patients = pd.DataFrame({
    "Age"                 : [34,  55,  28,  67,  45,  72,  38,  60,  25,  50],
    "Gender"              : ["Female","Male","Male","Female","Male","Male","Female","Male","Female","Male"],
    "Blood Pressure"      : [112, 168, 118, 172, 145, 178, 120, 165, 110, 155],
    "Cholesterol Level"   : [162, 288, 158, 292, 235, 295, 170, 275, 155, 250],
    "Exercise Habits"     : ["High","Low","High","Low","Medium","Low","High","Low","High","Medium"],
    "Smoking"             : ["No","Yes","No","Yes","Yes","Yes","No","Yes","No","No"],
    "Family Heart Disease": ["No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes"],
    "Diabetes"            : ["No","Yes","No","Yes","No","Yes","No","Yes","No","No"],
    "BMI"                 : [19.2, 37.8, 21.5, 38.5, 29.5, 39.2, 20.1, 36.5, 18.8, 27.3],
    "High Blood Pressure" : ["No","Yes","No","Yes","Yes","Yes","No","Yes","No","No"],
    "Low HDL Cholesterol" : ["No","Yes","No","Yes","No","Yes","No","Yes","No","No"],
    "High LDL Cholesterol": ["No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes"],
    "Alcohol Consumption" : ["Low","High","Low","High","Medium","High","Low","High","Low","Low"],
    "Stress Level"        : ["Low","High","Low","High","Medium","High","Low","High","Low","Medium"],
    "Sleep Hours"         : [8.5,  4.2,  9.0,  4.0,  6.5,  4.1,  8.8,  4.5,  9.5,  7.0],
    "Sugar Consumption"   : ["Low","High","Low","High","Medium","High","Low","High","Low","Low"],
    "Triglyceride Level"  : [108, 392,  112, 396,  245, 398,  115, 385,  102, 210],
    "Fasting Blood Sugar" : [82,  158,   85, 159,  118, 160,   84, 155,   80, 105],
    "CRP Level"           : [0.6, 14.5,  0.8, 14.8,  7.2, 14.9,  0.5, 13.8,  0.4,  5.5],
    "Homocysteine Level"  : [5.5, 19.2,  6.0, 19.6,  12.5, 19.8,  5.8, 18.9,  5.2, 11.0],
})


# In[133]:


mp = multiple_patients.copy()

# Encoding
mp["Gender"] = (mp["Gender"] == "Male").astype(int)
for col in ["Smoking","Family Heart Disease","Diabetes",
            "High Blood Pressure","Low HDL Cholesterol","High LDL Cholesterol"]:
    mp[col] = (mp[col] == "Yes").astype(int)

for col in ["Exercise Habits","Stress Level","Sugar Consumption"]:
    mp[col] = mp[col].map({"Low":0,"Medium":1,"High":2})
mp["Alcohol Consumption"] = mp["Alcohol Consumption"].map({"Unknown":-1,"Low":0,"Medium":1,"High":2})

# Feature engineering
mp["BMI_BP_ratio"]         = mp["BMI"] / (mp["Blood Pressure"] + 1)
mp["Lipid_risk_score"]     = mp["Cholesterol Level"] + mp["Triglyceride Level"]
mp["Metabolic_risk_score"] = mp["Fasting Blood Sugar"] + mp["CRP Level"] + mp["Homocysteine Level"]

# Capping + Scaling
for col in clip_cols:
    mp[col] = mp[col].clip(*clip_bounds[col])
mp[scale_cols] = scaler.transform(mp[scale_cols])

# Predict
predictions   = xgb_model.predict(mp[feature_columns])
probabilities = xgb_model.predict_proba(mp[feature_columns])

results = multiple_patients.copy()
results["Prediction"]        = ["❤️ Disease" if p==1 else "✅ No Disease" for p in predictions]
results["Disease Prob %"]    = (probabilities[:,1] * 100).round(2)
results["No Disease Prob %"] = (probabilities[:,0] * 100).round(2)
results["Risk Level"]        = pd.cut(probabilities[:,1],
                                       bins=[0,0.3,0.6,1.0],
                                       labels=["🟢 Low","🟡 Medium","🔴 High"])

print("=" * 65)
print("            PREDICTION RESULTS")
print("=" * 65)
print(results[["Age","Gender","Smoking","Diabetes",
               "Prediction","Disease Prob %",
               "No Disease Prob %","Risk Level"]].to_string(index=False))


# In[ ]:




