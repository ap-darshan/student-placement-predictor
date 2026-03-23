import streamlit as st
import pickle
import numpy as np

# Load model and features
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.title("🎓 Student Placement Predictor")

# Inputs
cgpa = st.slider("CGPA", 0.0, 10.0)
internships = st.number_input("Internships", 0, 5)
projects = st.number_input("Projects", 0, 10)
coding = st.slider("Coding Skills", 1, 10)
communication = st.slider("Communication Skills", 1, 10)
aptitude = st.slider("Aptitude Score", 0, 100)
soft_skills = st.slider("Soft Skills Rating", 1, 10)
certifications = st.number_input("Certifications", 0, 10)
backlogs = st.number_input("Backlogs", 0, 5)

import pandas as pd

if st.button("Predict"):
    input_dict = {
        "Age": 22,
        "CGPA": cgpa,
        "Internships": internships,
        "Projects": projects,
        "Coding_Skills": coding,
        "Communication_Skills": communication,
        "Aptitude_Test_Score": aptitude,
        "Soft_Skills_Rating": soft_skills,
        "Certifications": certifications,
        "Backlogs": backlogs,

        # Default categorical values
        "Gender": "Male",
        "Degree": "B.Tech",
        "Branch": "CSE"
    }

    input_df = pd.DataFrame([input_dict])

    # Apply encoding
    input_df = pd.get_dummies(input_df)

    # Align with training columns
    input_df = input_df.reindex(columns=features, fill_value=0)

    result = model.predict(input_df)

    if result[0] == 1:
        st.success("✅ Likely to be Placed")
    else:
        st.error("❌ Not Likely to be Placed")