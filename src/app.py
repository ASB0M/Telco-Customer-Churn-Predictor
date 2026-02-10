import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Add src to path to import data_processing if running from root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)

try:
    from data_processing import preprocess_data
except ImportError:
    # If standard import fails, try relative import or checking sys.path
    import sys
    sys.path.append(os.path.join(parent_dir, 'src'))
    from data_processing import preprocess_data

# Set page config
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join(parent_dir, "models", "xgb_model.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please run train.py first.")
        return None
    return joblib.load(model_path)

model = load_model()

# Header
st.title("🔮 Telco Customer Churn Predictor")
st.markdown("""
This app predicts the likelihood of a customer churning based on their subscription details.
Adjust the values in the sidebar to see the prediction updates in real-time.
""")

# Sidebar Inputs
st.sidebar.header("Customer Profile")

def user_input_features():
    # Categorical Inputs
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ("No", "Yes"))
    partner = st.sidebar.selectbox("Partner", ("No", "Yes"))
    dependents = st.sidebar.selectbox("Dependents", ("No", "Yes"))
    
    # Service Inputs
    phone_service = st.sidebar.selectbox("Phone Service", ("No", "Yes"))
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ("No phone service", "No", "Yes"))
    internet_service = st.sidebar.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    online_security = st.sidebar.selectbox("Online Security", ("No internet service", "No", "Yes"))
    online_backup = st.sidebar.selectbox("Online Backup", ("No internet service", "No", "Yes"))
    device_protection = st.sidebar.selectbox("Device Protection", ("No internet service", "No", "Yes"))
    tech_support = st.sidebar.selectbox("Tech Support", ("No internet service", "No", "Yes"))
    streaming_tv = st.sidebar.selectbox("Streaming TV", ("No internet service", "No", "Yes"))
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ("No internet service", "No", "Yes"))
    
    # Contract & Billing
    contract = st.sidebar.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ("No", "Yes"))
    payment_method = st.sidebar.selectbox("Payment Method", (
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ))
    
    # Numerical Inputs
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 1)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=monthly_charges * tenure)

    # Dictionary for DataFrame
    data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main Panel
st.subheader("Customer Data")
st.write(input_df)

if st.button("Predict Churn"):
    if model:
        # Preprocess input
        processed_df = preprocess_data(input_df)
        
        # Predict
        prediction = model.predict(processed_df)
        probability = model.predict_proba(processed_df)
        
        churn_prob = probability[0][1]
        is_churn = prediction[0] == 1
        
        st.subheader("Prediction Result")
        
        # Display simplified result
        if is_churn:
            st.error(f"**High Risk of Churn** ({churn_prob:.1%})")
            st.write("This customer is likely to churn.")
        else:
            st.success(f"**Low Risk of Churn** ({churn_prob:.1%})")
            st.write("This customer is likely to stay.")
            
        # Optional: Visualization of probability
        st.progress(float(churn_prob))
    else:
        st.error("Model could not be loaded. Check logs.")

st.markdown("---")
st.markdown("Built with Streamlit & XGBoost")
