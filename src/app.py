import streamlit as st

st.set_page_config(page_title="Automobile Customer Segmentation", layout="wide")

st.title("Customer Segmentation Classification")

col1, col2, col3 = st.columns(3, gap="medium")

# Get user inputs

with col1:
    age = st.text_input("Age", "Enter your age here")
    married = st.text_input("Married (yes/no)", "Enter yes or no here")
    work_experience = st.text_input("Work Experience", "Enter your work experience here")

with col2:
    graduated = st.text_input("Graduated (yes/no)", "Enter yes or no here")
    profession = st.text_input("Profession", "Enter your profession here")

with col3:
    spending_score = st.text_input("Spending Score", "Enter your spending score here")
    family_size = st.text_input("Family Size", "Enter your family size here")

if st.button("Submit"):
    try:
        age = int(age)
    except ValueError:
        st.error("Age should be a number")

    if married not in ["yes", "no"]:
        st.error("Married should be either 'yes' or 'no'")
    elif married == "yes":
        married = 1
    else:
        married = 0

    if graduated not in ["yes", "no"]:
        st.error("Graduated should be either 'yes' or 'no'")
    elif graduated == "yes":
        graduated = 1
    else:
        graduated = 0

    try:
        spending_score = float(spending_score)
    except ValueError:
        st.error("Spending Score should be a number")

    try:
        family_size = float(family_size)
    except ValueError:
        st.error("Family Size should be a number")

    try:
        work_experience = float(work_experience)
    except ValueError:
        st.error("Work Experience should be a number")