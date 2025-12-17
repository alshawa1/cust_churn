import streamlit as st
import pandas as pd
import joblib
import os
import streamlit as st

st.write("📂 Current directory:", os.getcwd())
st.write("📄 Files here:", os.listdir())

# Load model & features
model = joblib.load("Telco_Churn_App/churn_model.pkl")
features = joblib.load("Telco_Churn_App/features.pkl")


st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction")

st.write("Enter customer details to predict churn")

# ========== Inputs ==========
tenure = st.number_input("Tenure (months)", 0, 100, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"]
)

# ========== Encoding ==========
input_data = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,
    "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
    "InternetService_No": 1 if internet == "No" else 0,
    "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
    "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,
    "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0
}])

# ترتيب الأعمدة زي التدريب
input_data = input_data.reindex(columns=features, fill_value=0)

# ========== Prediction ==========
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Customer is likely to STAY\nProbability: {probability:.2%}")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "churn_model.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
st.write("Model coefficients:", model.coef_)

