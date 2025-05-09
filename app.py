#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 15:58:37 2025

@author: Group13
"""

import streamlit as st
import pickle
import numpy as np

#Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Diabetes Prediction by Group 13")

#Example input fields – update according to your model's expected features
st.subheader("Input Features")
Glucose = st.number_input("Glucose", value=0.0)
BMI = st.number_input("BMI", value=0.0)
BloodPressure = st.number_input("BloodPressure", value=0.0)
Insulin = st.number_input("Insulin", value=0.0)
Age = st.number_input("Age", value=0.0)


if st.button("Predict"):
    features = np.array([[Glucose, BMI, BloodPressure, Insulin, Age]])
    prediction = model.predict(features)
    result = "Yes, you have diabetes" if prediction[0] == 1 else "No, you don't have diabetes"
    st.success(f"{result}")
