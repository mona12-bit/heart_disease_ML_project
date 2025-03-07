import streamlit as st
import joblib
import numpy as np
import os

# Correct model path
model_path = "/kaggle/working/random_forest_model.pkl"

# Load model only if it exists
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.write("✅ Model loaded successfully!")
else:
    st.error(f"❌ Model file not found at {model_path}. Please check the path.")

# Streamlit UI
st.title("Heart Disease Prediction")
st.write("Enter the patient details below:")

age = st.number_input("Age", min_value=20, max_value=100, value=50)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
bp = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)

if st.button("Predict"):
    if os.path.exists(model_path):
        input_data = np.array([[age, cholesterol, bp]])
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("The model predicts that the patient has heart disease.")
        else:
            st.success("The model predicts that the patient is healthy.")
    else:
        st.error("Prediction failed. Model not found.")
