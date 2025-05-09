#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

st.title("🩺 Diabetes Prediction by Group 13")

st.subheader("Please input patient data:")

# Input fields for all 8 features
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=0.0, format="%.1f")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0, max_value=120, value=0)

# Format input for model
features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                      insulin, bmi, diabetes_pedigree, age]])

# Predict
if st.button("Predict"):
    prediction = model.predict(features)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.success(f"Prediction: {result}")
