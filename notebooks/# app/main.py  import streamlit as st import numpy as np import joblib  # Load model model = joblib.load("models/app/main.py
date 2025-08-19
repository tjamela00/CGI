# app/main.py

import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("models/match_predictor.pkl")

# App title
st.title("⚽ Match Win Probability Predictor")

# Inputs
home_adv = st.selectbox("Is the team playing at home?", ["Yes", "No"])
home_adv = 1 if home_adv == "Yes" else 0

injuries = st.slider("Number of Injured Key Players", 0, 10)
yellow_cards = st.slider("Number of Yellow Cards in Last Match", 0, 5)

# Predict
input_data = np.array([[home_adv, injuries, yellow_cards]])
proba = model.predict_proba(input_data)[0]

# Display results
st.subheader("Prediction Results")
st.write(f"🏆 **Win Probability**: {round(proba[2]*100, 2)}%")
st.write(f"🤝 **Draw Probability**: {round(proba[1]*100, 2)}%")
st.write(f"❌ **Loss Probability**: {round(proba[0]*100, 2)}%")
