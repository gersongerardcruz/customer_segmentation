import streamlit as st
from utils import *

st.set_page_config(page_title="Automobile Customer Segmentation", layout="wide")

st.title("Customer Segmentation Classification")

col1, col2, col3 = st.columns(3, gap="medium")

# List of features the model was trained on
features = ['Age', 'Married', 'Graduated', 'Gender', 'Profession', 'Spending Score', 'Family Size', 'Work Experience']

# Encoding mappings
encodings = {
    'Married': {'Yes': 1, 'No': 0},
    'Graduated': {'Yes': 1, 'No': 0},
    'Gender': {"Female": 0, "Male": 1},
    'Profession': {'Artist': 0, 'Doctor': 1, 'Engineer': 2, 'Entertainment': 3, 'Executive': 4, 'Healthcare': 5, "Lawyer": 6, "Other": 7},
    'Spending Score': {'Low': 2, 'Average': 0, 'High': 1}
}

# Get user inputs

with col1:
    age = st.text_input("Age", "Enter your age here")
    married = st.selectbox("Have you ever been married?", list(encodings["Married"].keys()))
    work_experience = st.text_input("Work Experience", "Enter your years of work experience here")

with col2:
    graduated = st.selectbox("Have you ever graduated?", list(encodings["Graduated"].keys()))
    profession = st.selectbox("What is your profession?", list(encodings["Profession"].keys()))
    gender =  st.selectbox("What is your sex?", list(encodings["Gender"].keys()))

with col3:
    spending_score = st.selectbox("How big of a spender on automobiles are you?", list(encodings["Spending Score"].keys()))
    family_size = st.text_input("Family Size", "Enter the number of family members you have here")
