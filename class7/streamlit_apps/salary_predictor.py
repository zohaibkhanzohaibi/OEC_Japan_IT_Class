# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 18:13:45 2025

@author: zebi2
"""

import streamlit as st
import pandas as pd
import joblib


# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("salary_model.pkl")  

model = load_model()

# Get feature names from model
feature_columns = model.feature_names_in_

# Streamlit UI
st.title("Salary Prediction System")
st.write("Enter the required feature values to predict the salary.")

# Create input fields dynamically
input_data = {}
for feature in feature_columns:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Predict Salary
if st.button("Predict Salary"):
    input_df = pd.DataFrame([input_data])  # Convert dictionary to DataFrame
    predicted_salary = model.predict(input_df)[0]
    st.success(f"Predicted Salary: {predicted_salary:,.2f}")