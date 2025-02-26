import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Check if the model file exists
model_path = "random_forest_model.pkl"  # Update if needed

if not os.path.exists(model_path):
    st.error(f"Model file not found at '{model_path}'. Please check the path and try again.")
else:
    # Load trained model
    model = joblib.load(model_path)

    # Title
    st.title("Heart Disease Prediction")

    # Input features
    st.write("Enter the patient details below:")

    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
    bp = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)

    # Make Prediction
    if st.button("Predict"):
        input_data = np.array([[age, cholesterol, bp]])  # Modify as per your dataset
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("The model predicts that the patient has heart disease.")
        else:
            st.success("The model predicts that the patient is healthy.")
