#!/usr/bin/env python3
-- coding: utf-8 --
"""
Created on Fri May  9 15:58:37 2025

@author: parichikaphumikakrak
"""

import streamlit as st
import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Diabetes Prediction by Group 13")

# Example input fields – update according to your model's expected features
st.subheader("Input Features")
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Collect input into an array (adjust shape as needed)
features = np.array([[feature1, feature2, feature3]])

if st.button("Predict"):
    prediction = model.predict(features)
    st.success(f"Prediction: {prediction[0]}")
