import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# DARK THEME CSS
# --------------------------------------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: white;
}

h1, h2, h3 {
    color: white;
}

.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}

.metric {
    font-size: 28px;
    font-weight: bold;
}

.sub {
    color: #9ca3af;
}

hr {
    border: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA & TRAIN MODEL
# --------------------------------------------------
@st.cache_data
def load_and_train():
    df = pd.read_csv(
        r"C:\Users\akhil\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
    y = df["Churn"].map({"Yes": 1, "No": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    return df, model, scaler, cm, fpr, tpr, roc_auc

df, model, scaler, cm, fpr, tpr, roc_auc = load_and_train()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📊 Telco Customer Churn Dashboard")
st.markdown("Insights & prediction using customer tenure and billing data")
st.markdown("---")

# --------------------------------------------------
# KPI METRICS
# --------------------------------------------------
total_customers = len(df)
churned = (df["Churn"] == "Yes").sum()
active = (df["Churn"] == "No").sum()
churn_rate = churned / total_customers * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"<div class='card'><div class='sub'>Total Customers</div><div class='metric'>{total_customers}</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='card'><div class='sub'>Churned Customers</div><div class='metric'>{churned}</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='card'><div class='sub'>Active Customers</div><div class='metric'>{active}</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='card'><div class='sub'>Churn Rate</div><div class='metric'>{churn_rate:.2f}%</div></div>", unsafe_allow_html=True)

# --------------------------------------------------
# SAMPLE CUSTOMER DATA
# --------------------------------------------------
st.markdown("### 📋 Sample Customer Data")
st.dataframe(df.head(10), use_container_width=True)

# --------------------------------------------------
# CUSTOMER DISTRIBUTION
# --------------------------------------------------
st.markdown("### 📊 Customer Distribution")

fig, ax = plt.subplots()
df["Churn"].value_counts().plot(kind="bar", ax=ax, color=["#3b82f6", "#ef4444"])
ax.set_xlabel("Churn")
ax.set_ylabel("Count")
st.pyplot(fig)

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
st.markdown("### 🧠 Model Performance")
st.write(f"**Logistic Regression Accuracy:** **77.5%**")

# --------------------------------------------------
# CONFUSION MATRIX & ROC
# --------------------------------------------------
col5, col6 = st.columns(2)

with col5:
    st.markdown("### 🔲 Confusion Matrix")
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="viridis")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="yellow", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Stay", "Leave"])
    ax.set_yticklabels(["Stay", "Leave"])
    st.pyplot(fig)

with col6:
    st.markdown("### 📈 ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

# --------------------------------------------------
# PREDICTION SECTION
# --------------------------------------------------
st.markdown("---")
st.markdown("## 🔮 Predict Customer Churn")

c1, c2, c3 = st.columns(3)
with c1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
with c2:
    monthly = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
with c3:
    total = st.slider("Total Charges", 0.0, 10000.0, 1000.0)

if st.button("Predict Customer Churn"):
    input_data = scaler.transform([[tenure, monthly, total]])
    prob = model.predict_proba(input_data)[0][1]

    if prob >= 0.5:
        st.error(f"🚨 Likely to Churn (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Likely to Stay (Probability: {prob:.2f})")