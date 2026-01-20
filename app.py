import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="J.P. Morgan Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🏦 Credit Risk Analysis Dashboard")
st.markdown(
    "Built after completing **J.P. Morgan Quantitative Research Virtual Experience (Forage)**"
)

# ---------------- DATA GENERATION ----------------
np.random.seed(42)
data = pd.DataFrame({
    "Age": np.random.randint(21, 65, 500),
    "Income": np.random.randint(20000, 120000, 500),
    "Loan_Amount": np.random.randint(5000, 50000, 500),
    "Credit_Score": np.random.randint(300, 850, 500),
    "Default": np.random.choice([0, 1], size=500, p=[0.75, 0.25])
})

# ---------------- SIDEBAR ----------------
st.sidebar.header("📊 Filter Options")
min_score = st.sidebar.slider("Minimum Credit Score", 300, 850, 300)
filtered_data = data[data["Credit_Score"] >= min_score]

# ---------------- KPI METRICS ----------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Loans", len(filtered_data))
col2.metric("Average Credit Score", int(filtered_data["Credit_Score"].mean()))
col3.metric("Default Rate", f"{filtered_data['Default'].mean()*100:.2f}%")
col4.metric("Average Loan Amount", f"${int(filtered_data['Loan_Amount'].mean())}")

# ---------------- VISUALIZATIONS ----------------
st.subheader("📈 Credit Score Distribution")
fig1 = px.histogram(
    filtered_data,
    x="Credit_Score",
    nbins=30,
    title="Distribution of Credit Scores"
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("💸 Loan Amount vs Credit Score")
fig2 = px.scatter(
    filtered_data,
    x="Credit_Score",
    y="Loan_Amount",
    color="Default",
    title="Loan Amount vs Credit Score"
)
st.plotly_chart(fig2, use_container_width=True)

# ---------------- MODEL ----------------
st.subheader("🤖 Credit Default Prediction Model")

X = data[["Age", "Income", "Loan_Amount", "Credit_Score"]]
y = data["Default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"Model Accuracy: **{accuracy*100:.2f}%**")

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    columns=["Predicted No Default", "Predicted Default"],
    index=["Actual No Default", "Actual Default"]
)

st.write("Confusion Matrix")
st.dataframe(cm_df)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("📌 *Educational project inspired by J.P. Morgan Quantitative Research Simulation*")

import streamlit as st

st.set_page_config(page_title="Test App")

st.title("✅ Streamlit is Working!")
st.write("If you can see this page, your setup is correct 🎉")

