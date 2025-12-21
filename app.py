"""
Streamlit Application for Insurance Sales Prediction
Predicts whether a customer will respond positively to an insurance offer.
"""

import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Page configuration
st.set_page_config(
    page_title="Insurance Sales Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 8.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 1rem;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_and_load_model():
    """Train the model if not exists, then load it."""
    model_path = 'models/model.pkl'
    
    # If model exists, load it
    if os.path.exists(model_path):
        try:
            with open('models/model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            with open('models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            
            with open('models/label_encoders.pkl', 'rb') as f:
                label_encoders = pickle.load(f)
            
            return model, scaler, label_encoders
        except Exception as e:
            st.warning(f"Error loading model: {e}. Retraining...")
    
    # Train model if it doesn't exist
    with st.spinner("Training model... This may take a moment."):
        # Load dataset
        df = pd.read_csv("Dataset.csv")
        
        # Drop ID column
        df.drop(columns=['id'], inplace=True)
        
        # Encode categorical columns
        categorical_cols = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Split features and target
        X = df.drop('Response', axis=1)
        y = df['Response']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Logistic Regression model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save model and preprocessors
        os.makedirs('models', exist_ok=True)
        
        with open('models/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        with open('models/label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
    
    return model, scaler, label_encoders

def preprocess_input(data, label_encoders):
    """Preprocess input data using saved label encoders."""
    df = data.copy()
    
    # Encode categorical columns
    categorical_cols = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
    for col in categorical_cols:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
    
    return df

def main():
    # Header
    st.markdown('<p class="main-header">📊 Insurance Sales Prediction</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load or train model
    model, scaler, label_encoders = train_and_load_model()
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("ℹ️ Instructions")
        st.markdown("""
        Fill in the customer information below to predict 
        whether they will respond positively to an insurance offer.
        
        **Response:**
        - **YES**: Customer is likely to purchase insurance
        - **NO**: Customer is unlikely to purchase insurance
        """)
        st.markdown("---")
        st.markdown("### Model Information")
        st.info("Model: Logistic Regression")
        st.info("Accuracy: ~87.7%")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Information")
        
        # Input fields
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        driving_license = st.selectbox("Driving License", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        region_code = st.number_input("Region Code", min_value=0.0, max_value=100.0, value=28.0, step=1.0)
        previously_insured = st.selectbox("Previously Insured", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
    with col2:
        st.subheader("Vehicle Information")
        
        vehicle_age = st.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])
        vehicle_damage = st.selectbox("Vehicle Damage", ["Yes", "No"])
        annual_premium = st.number_input("Annual Premium", min_value=0.0, value=35000.0, step=1000.0, format="%.2f")
        policy_sales_channel = st.number_input("Policy Sales Channel", min_value=0.0, max_value=200.0, value=152.0, step=1.0)
        vintage = st.number_input("Vintage (days)", min_value=0, max_value=500, value=120, step=1)
    
    # Prediction button
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predict_button = st.button("🔮 Predict Insurance Response", type="primary", use_container_width=True)
    
    # Make prediction
    if predict_button:
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Age': [age],
                'Driving_License': [driving_license],
                'Region_Code': [region_code],
                'Previously_Insured': [previously_insured],
                'Vehicle_Age': [vehicle_age],
                'Vehicle_Damage': [vehicle_damage],
                'Annual_Premium': [annual_premium],
                'Policy_Sales_Channel': [policy_sales_channel],
                'Vintage': [vintage]
            })
            
            # Preprocess input
            processed_data = preprocess_input(input_data, label_encoders)
            
            # Scale features
            scaled_data = scaler.transform(processed_data)
            
            # Make prediction
            prediction = model.predict(scaled_data)[0]
            prediction_proba = model.predict_proba(scaled_data)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("### Prediction Results")
            
            result_col1, result_col2 = st.columns([1, 1])
            
            with result_col1:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                if prediction == 1:
                    st.markdown('<p class="positive">✅ Response: YES</p>', unsafe_allow_html=True)
                    st.success("Customer is likely to purchase insurance!")
                else:
                    st.markdown('<p class="negative">❌ Response: NO</p>', unsafe_allow_html=True)
                    st.warning("Customer is unlikely to purchase insurance.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with result_col2:
                st.markdown("**Prediction Probabilities:**")
                prob_no = prediction_proba[0] * 100
                prob_yes = prediction_proba[1] * 100
                
                st.metric("Probability of NO", f"{prob_no:.2f}%")
                st.metric("Probability of YES", f"{prob_yes:.2f}%")
                
                # Progress bars
                st.progress(prob_no / 100, text=f"NO: {prob_no:.1f}%")
                st.progress(prob_yes / 100, text=f"YES: {prob_yes:.1f}%")
            
            # Display input summary
            with st.expander("📋 View Input Summary"):
                st.dataframe(input_data, use_container_width=True)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()

