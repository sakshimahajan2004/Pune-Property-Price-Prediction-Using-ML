"""
app.py
------
Streamlit web app for Pune Property Price Prediction.
Uses trained ML model and feature alignment from model_columns.pkl.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------------------------------
# 1. Load model and columns
# -----------------------------------------------------
@st.cache_resource
def load_model():
    model_file = "best_model_RandomForest.pkl"
    columns_file = "model_columns.pkl"

    if not os.path.exists(model_file):
        st.error("❌ Model file not found. Please train the model first.")
        st.stop()
    if not os.path.exists(columns_file):
        st.error("❌ model_columns.pkl not found. Please run model_built.py first.")
        st.stop()

    model = joblib.load(model_file)
    model_columns = joblib.load(columns_file)
    return model, model_columns


# -----------------------------------------------------
# 2. Prepare input data
# -----------------------------------------------------
def prepare_input(user_inputs, model_columns):
    """
    Convert user input into the same format used during training
    (with correct one-hot encoded columns).
    """
    df = pd.DataFrame([user_inputs])

    # Add missing columns
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Drop any extras and order columns correctly
    df = df[model_columns]
    return df


# -----------------------------------------------------
# 3. Streamlit UI
# -----------------------------------------------------
def main():
    st.set_page_config(page_title="🏠 Pune Property Price Predictor", layout="wide")

    st.title("🏠 Pune Property Price Prediction App")
    st.markdown("Use this app to estimate **property prices** in Pune based on various factors.")

    st.sidebar.header("🔧 Input Property Details")

    # Sidebar inputs
    bhk = st.sidebar.number_input("BHK (Bedrooms)", 1, 10, 2)
    area = st.sidebar.number_input("Built-up Area (sq. ft.)", 200, 10000, 1000)
    bathrooms = st.sidebar.number_input("Number of Bathrooms", 1, 10, 2)
    balcony = st.sidebar.selectbox("Balcony", [0, 1, 2, 3])
    furnishing = st.sidebar.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully Furnished"])
    location = st.sidebar.text_input("Location", "Kothrud")
    age = st.sidebar.selectbox("Property Age", ["New", "1-5 Years", "5-10 Years", "10+ Years"])
    amenities = st.sidebar.multiselect("Amenities", ["Lift", "Parking", "Gym", "Pool", "Security", "Garden"])
    additionalrooms = st.sidebar.selectbox("Additional Rooms", ["None", "Study Room", "Pooja Room", "Servant Room"])

    # Load model and columns
    model, model_columns = load_model()

    # Predict button
    if st.button("💰 Predict Property Price"):
        user_data = {
            "bhk": bhk,
            "area": area,
            "bathroom": bathrooms,
            "balcony": balcony,
            "furnishing": furnishing,
            "location": location,
            "age": age,
            "amenities": ", ".join(amenities),
            "additionalrooms": additionalrooms
        }

        # Align with model training columns
        input_df = prepare_input(user_data, model_columns)

        # Predict
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"🏡 Estimated Property Price: **₹ {prediction:,.2f}**")
        except Exception as e:
            st.error(f"❗ Prediction failed: {e}")


# -----------------------------------------------------
# Run the app
# -----------------------------------------------------
if __name__ == "__main__":
    main()
