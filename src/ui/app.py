# src/ui/app.py
import streamlit as st
import requests
import pandas as pd
import json

# Page config
st.set_page_config(
    page_title="Real-Time Credit Card Fraud Detector",
    page_icon="🔒",
    layout="centered"
)

st.title("🔒 Real-Time Credit Card Fraud Detection System")
st.markdown("A system that detects fraudulent transactions in streaming data, using anomaly detection.")

# API URL (change to your deployed URL later)
API_URL = "http://localhost:8000"  # For local testing
# When deployed, change to your public URL, e.g., https://your-api.onrender.com

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    mode = st.radio("Select Mode", ["Single Transaction", "Batch Upload (CSV)"])
    st.markdown("---")
    st.info("Processes live data feeds, utilizes machine learning models to flag suspicious transactions")

if mode == "Single Transaction":
    st.subheader("Enter Transaction Details")
    
    col1, col2 = st.columns(2)
    with col1:
        time = st.number_input("Time (seconds from first)", value=0.0)
        amount = st.number_input("Amount", value=149.62, format="%.2f")
        v1 = st.number_input("V1", value=-1.359807)
        v2 = st.number_input("V2", value=-0.072781)
        v3 = st.number_input("V3", value=2.536347)
        v4 = st.number_input("V4", value=1.378155)
    with col2:
        v5 = st.number_input("V5", value=-0.338321)
        v6 = st.number_input("V6", value=0.462388)
        v7 = st.number_input("V7", value=0.239599)
        v8 = st.number_input("V8", value=0.098698)
        v9 = st.number_input("V9", value=0.363787)
        v10 = st.number_input("V10", value=0.090794)

    # More features (compact)
    with st.expander("More Features (V11-V28)"):
        col3, col4 = st.columns(2)
        with col3:
            v11 = st.number_input("V11", value=-0.551600)
            v12 = st.number_input("V12", value=-0.617801)
            v13 = st.number_input("V13", value=-0.991390)
            v14 = st.number_input("V14", value=-0.311169)
            v15 = st.number_input("V15", value=1.468177)
            v16 = st.number_input("V16", value=-0.470401)
        with col4:
            v17 = st.number_input("V17", value=0.207971)
            v18 = st.number_input("V18", value=0.025791)
            v19 = st.number_input("V19", value=0.403993)
            v20 = st.number_input("V20", value=0.251412)
            v21 = st.number_input("V21", value=-0.018307)
            v22 = st.number_input("V22", value=0.277838)
            v23 = st.number_input("V23", value=-0.110474)
            v24 = st.number_input("V24", value=0.066928)
            v25 = st.number_input("V25", value=0.128539)
            v26 = st.number_input("V26", value=-0.189115)
            v27 = st.number_input("V27", value=0.133558)
            v28 = st.number_input("V28", value=-0.021053)

    if st.button("🔍 Detect Fraud", type="primary"):
        payload = {
            "Time": float(time),
            "Amount": float(amount),
            "V1": float(v1), "V2": float(v2), "V3": float(v3), "V4": float(v4),
            "V5": float(v5), "V6": float(v6), "V7": float(v7), "V8": float(v8),
            "V9": float(v9), "V10": float(v10),
            "V11": float(v11), "V12": float(v12), "V13": float(v13), "V14": float(v14),
            "V15": float(v15), "V16": float(v16), "V17": float(v17), "V18": float(v18),
            "V19": float(v19), "V20": float(v20), "V21": float(v21), "V22": float(v22),
            "V23": float(v23), "V24": float(v24), "V25": float(v25), "V26": float(v26),
            "V27": float(v27), "V28": float(v28)
        }

        with st.spinner("Analyzing transaction..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                result = response.json()

                prob = result["fraud_probability"]
                is_fraud = result["is_fraud"]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fraud Probability", f"{prob:.2f}%")
                with col2:
                    st.metric("Risk Level", "HIGH" if is_fraud else "LOW")
                with col3:
                    st.metric("Decision", "🚨 FRAUD" if is_fraud else "✅ SAFE")

                if is_fraud:
                    st.error("⚠️ High risk of fraud detected!")
                else:
                    st.success("Transaction appears legitimate.")

            except Exception as e:
                st.error(f"API Error: {str(e)}")

elif mode == "Batch Upload (CSV)":
    st.subheader("Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())

        if st.button("Process Batch"):
            records = df.to_dict(orient="records")
            with st.spinner(f"Processing {len(records)} transactions..."):
                try:
                    response = requests.post(f"{API_URL}/predict-batch", json=records)
                    results = response.json()["predictions"]

                    result_df = pd.DataFrame(results)
                    result_df.index = df.index
                    result_df["original_amount"] = df["Amount"]

                    fraud_count = result_df["is_fraud"].sum()
                    st.write(f"### Results: {fraud_count} fraudulent transactions detected")

                    st.dataframe(result_df.style.highlight_max(axis=0))

                    csv = result_df.to_csv().encode()
                    st.download_button("Download Results", csv, "fraud_predictions.csv", "text/csv")

                except Exception as e:
                    st.error(f"Batch prediction failed: {str(e)}")

# Footer
st.markdown("---")
st.caption("Real-Time Fraud Detection System • End-to-End ML Project")