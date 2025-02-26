import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

uploaded_file = st.file_uploader("Upload the trained model (random_forest_model.pkl)", type="pkl")

if uploaded_file is not None:
    model = joblib.load(uploaded_file)
    st.success("Model loaded successfully!")
else:
    st.error("Please upload 'random_forest_model.pkl'.")

# Title
st.title("Heart Disease Prediction App")

# Input features
st.write("Enter the patient details below:")

# Example feature inputs (modify according to your dataset)
age = st.number_input("Age", min_value=20, max_value=100, value=50)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
bp = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)

# Additional inputs if your model needs more features (modify as needed)
# Example: st.number_input("Feature_Name", min_value=0, max_value=100, value=50)

# Make Prediction
if st.button("Predict"):
    try:
        # Modify this based on the number and order of features in your trained model
        input_data = np.array([[age, cholesterol, bp]])

        # Ensure the feature count matches the trained model
        expected_features = model.n_features_in_
        if input_data.shape[1] != expected_features:
            st.error(f"Expected {expected_features} input features, but got {input_data.shape[1]}. Check input format.")
        else:
            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.error("The model predicts that the patient has heart disease.")
            else:
                st.success("The model predicts that the patient is healthy.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
