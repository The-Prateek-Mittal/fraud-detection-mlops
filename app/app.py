import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Fraud Detector", page_icon="🔍", layout="wide")

APP_MODE = os.environ.get("APP_MODE", "local")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.45"))

@st.cache_resource
def load_model():
    if APP_MODE == "mlflow":
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.xgboost.load_model("models:/FraudDetectionXGB/Production")
        scaler = joblib.load("scaler.pkl")
    else:  # local mode for Streamlit Cloud
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model("models/model.json")
        scaler = joblib.load("models/scaler.pkl")
    return model, scaler

def main():
    st.title("Fraud Detector 🔍")
    st.write("Check credit card transactions for potential fraud.")

    if APP_MODE == "mlflow":
        st.info("Running with MLflow model registry")
    else:
        st.info("Running with locally saved model (Streamlit Cloud mode)")

    model, scaler = load_model()

    st.sidebar.header("Transaction Features")
    amount = st.sidebar.number_input("Transaction Amount", min_value=0.1, max_value=100000.0, value=500.0)
    hour = st.sidebar.slider("Hour of day", min_value=0, max_value=23, value=14)
    
    st.sidebar.markdown("### V-Features")
    v_features = {}
    for i in range(1, 29):
        v_features[f"V{i}"] = st.sidebar.number_input(f"V{i}", min_value=-10.0, max_value=10.0, value=0.0)

    if st.button("Check Transaction"):
        st.progress(0)
        
        amount_log = np.log1p(amount)
        hour_float = float(hour)
        
        # Build input DataFrame matching the exact training columns order
        # V1-V28, Amount_log, Hour
        cols = [f"V{i}" for i in range(1, 29)] + ["Amount_log", "Hour"]
        row = [v_features[f"V{i}"] for i in range(1, 29)] + [amount_log, hour_float]
        df_input = pd.DataFrame([row], columns=cols)
        
        # Scale Amount_log and Hour
        df_input[['Amount_log', 'Hour']] = scaler.transform(df_input[['Amount_log', 'Hour']])
        
        st.progress(50)
        
        # Predict
        prob = model.predict_proba(df_input)[0, 1]
        
        st.progress(100)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud Probability", f"{prob:.4f}")
        col2.metric("Decision Threshold", f"{FRAUD_THRESHOLD:.4f}")
        
        with col3:
            st.markdown("### Verdict")
            if prob >= FRAUD_THRESHOLD:
                st.markdown("<h3 style='color: red;'>FRAUD DETECTED — Block this transaction!</h3>", unsafe_allow_html=True)
                st.error("FRAUD DETECTED — Block this transaction!")
            else:
                st.markdown("<h3 style='color: green;'>SAFE</h3>", unsafe_allow_html=True)
                st.success("Transaction looks safe. Approve it.")

    st.markdown("---")
    st.markdown("Model: XGBoost | Tracking: MLflow | Deployed via: Streamlit Cloud + Minikube")

if __name__ == "__main__":
    main()
