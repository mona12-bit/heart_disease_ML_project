import streamlit as st
import numpy as np
import joblib
import os

# Define correct path
model_path = "/kaggle/working/random_forest_model.pkl"  

# Try loading the model
model = None  # Initialize model variable
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
else:
    st.error(f"❌ Error: Model file not found at {model_path}. Please upload it.")

# Streamlit UI starts here
st.title("Heart Disease Prediction")
st.write("Enter the patient details below:")

# Input features
age = st.number_input("Age", min_value=20, max_value=100, value=50)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
bp = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)

# Prediction button
if st.button("Predict"):
    if model is not None:
        input_data = np.array([[age, cholesterol, bp]])  # Ensure correct input format
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("The model predicts that the patient has heart disease.")
        else:
            st.success("The model predicts that the patient is healthy.")
    else:
        st.error("⚠ Model not loaded! Please check the file path or upload the model.")
