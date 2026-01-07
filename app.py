import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ------------------ CUSTOM STYLE ------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
.title {
    text-align: center;
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown("<h1 class='title'>📊 Customer Churn Prediction</h1>", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
df = pd.read_csv("yash.csv")

# Select required columns
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']

# Clean TotalCharges
X['TotalCharges'] = X['TotalCharges'].replace(' ', np.nan)
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'])
X = X.dropna()
y = y.loc[X.index]

# Encode target
y = y.map({'Yes': 1, 'No': 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ------------------ USER INPUT ------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔍 Enter Customer Details")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, value=800.0)

if st.button("Predict Churn"):
    user_data = np.array([[tenure, monthly, total]])
    user_data_scaled = scaler.transform(user_data)

    prediction = model.predict(user_data_scaled)[0]
    probability = model.predict_proba(user_data_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Customer is likely to STAY\n\nProbability: {probability:.2f}")

st.markdown("</div>", unsafe_allow_html=True)

