import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ✅ Correct path to the model (since it's in the working directory)
MODEL_PATH = "/kaggle/working/random_forest_model.pkl"

# ✅ Check if the model file exists before loading
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at '{MODEL_PATH}'. Please check the path and try again.")
else:
    model = joblib.load(MODEL_PATH)

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
