import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.predict_wrapper import ChurnPredictor
import joblib

# Page Config
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# Load Logic
@st.cache_resource
def get_predictor():
    return ChurnPredictor()

predictor = get_predictor()

# UI Header
st.title("📡 Telco Customer Churn Predictor")
st.markdown("""
This application uses a **Tuned Random Forest Model** to predict the likelihood of a customer leaving.
Fill in the customer details below to get a real-time prediction.
""")

# Layout: Split into Columns
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📝 Customer Profile")
    
    # Input Form
    with st.form("churn_form"):
        st.subheader("Demographics")
        senior = st.checkbox("Senior Citizen")
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
        
        st.subheader("Services")
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        st.subheader("Account")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)
        
        submitted = st.form_submit_button("Predict Churn Risk")

with col2:
    if submitted:
        input_data = {
            'SeniorCitizen': 1 if senior else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'InternetService': internet,
            'Contract': contract,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly,
            'TotalCharges': total
        }
        
        # Make Prediction
        result = predictor.predict(input_data)
        prob = result['probability'] * 100
        churn_class = result['churn_prediction']
        
        st.divider()
        st.subheader("📊 Prediction Results")
        
        # Visualizing Prob
        if prob > 50:
            st.error(f"⚠️ High Churn Risk: {prob:.2f}%")
        else:
            st.success(f"✅ Low Churn Risk: {prob:.2f}%")
            
        st.progress(int(prob))
        
        # Contextual Advice
        st.write("### 💡 Recommended Actions")
        if prob > 70:
            st.warning("Action: Offer a 15% discount on 1-year contract immediately.")
        elif prob > 50:
             st.info("Action: Schedule a customer success call.")
        else:
             st.success("Action: No immediate action needed. Customer is healthy.")

    else:
        # Default View - Show Feature Importance
        st.write("### Model Insights")
        st.info("👈 Enter customer details on the left to see predictions.")
        st.image("plots/feature_importance.png", caption="Top Factors Driving Churn", use_column_width=True)
