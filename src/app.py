import streamlit as st
import numpy as np
import joblib
import json
from utils import *

st.set_page_config(page_title="Automobile Customer Segmentation", layout="wide")

# Add project descriptions
st.title("Customer Segmentation Classification")
st.markdown("This is a Streamlit deployment of a customer segmentation model trained on [Kaggle's Customer Segmentation Classification Dataset](https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation?select=Train.csv).")
st.markdown("The goal of the project is to classify the customer based on four segments: A, B, C, and D. These segments are described as follows: ") 

with open("../notebooks/descriptions.json") as f:
    segment_descriptions = json.load(f)

segment_descriptions = pd.DataFrame(segment_descriptions.values(), index=segment_descriptions.keys(), columns=["description"])
st.table(segment_descriptions)

st.markdown("These descriptions are **by no means definitive**. They are simply the result of Exploratory Data Analysis performed on this [notebook](https://github.com/gersongerardcruz/customer_segmentation/blob/main/notebooks/eda_on_segments.ipynb). Feel free to check it out!")

st.subheader("User Input for Segment Prediction")

col1, col2, col3 = st.columns(3)

# List of features and column names the model was trained on in order
num_features = ['Family Size', 'Age', 'Work Experience']
cat_features = ['Spending Score', 'Profession', 'Gender', 'Graduated', 'Married']
columns = ['Family_Size', 'Age', 'Work_Experience', 'Spending_Score',
       'Profession_Artist', 'Profession_Doctor', 'Profession_Engineer',
       'Profession_Entertainment', 'Profession_Executive',
       'Profession_Healthcare', 'Profession_Lawyer', 'Profession_Other',
       'Gender', 'Graduated', 'Ever_Married']

# Encoding mappings
encodings = {
    'Married': {'Yes': 1, 'No': 0},
    'Graduated': {'Yes': 1, 'No': 0},
    'Gender': {"Female": 0, "Male": 1},
    'Profession': {'Artist': 0, 'Doctor': 1, 'Engineer': 2, 'Entertainment': 3, 'Executive': 4, 'Healthcare': 5, "Lawyer": 6, "Other": 7},
    'Spending Score': {'Low': 2, 'Average': 0, 'High': 1}
}

# Load scaler for numerical values
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = np.load('minmax_scaler_params.npy')

# Set up Streamlit structure
with col1:
    age = st.number_input("Age")
    married = st.selectbox("Have you ever been married?", list(encodings["Married"].keys()))
    work_experience = st.number_input("Work Experience")

with col2:
    graduated = st.selectbox("Have you ever graduated?", list(encodings["Graduated"].keys()))
    work_profession = st.selectbox("What is your profession?", list(encodings["Profession"].keys()))
    gender =  st.selectbox("What is your sex?", list(encodings["Gender"].keys()))

with col3:
    spending_score = st.selectbox("How big of a spender on automobiles are you?", list(encodings["Spending Score"].keys()))
    family_size = st.number_input("Family Size")

# Get user inputs with the same order as the train data columns
inputs = {'Family Size': family_size, 'Age': age, 'Work Experience': work_experience, 'Spending Score': spending_score, 
          'Profession': work_profession, 'Gender':gender, 'Graduated': graduated, 'Married':married 
          }

model = joblib.load("../models/model.joblib")

if st.button("Submit"):
    # Preprocess the inputs
    # create a dictionary with only the numerical inputs
    num_inputs = {k: v for k, v in inputs.items() if k in num_features}

    # convert the numerical inputs to a dataframe
    num_df = pd.DataFrame.from_dict(num_inputs, orient='index').T

    # scale numerical columns
    scaled_inputs = scaler.transform(num_df)

    # create a dataframe for the scaled inputs
    num_df = pd.DataFrame(scaled_inputs)

    # Create a numpy array of zeroes equal to the number of encoded features
    # and the number of onehot encoded features
    num_professions = len(set(encodings['Profession'].values()))
    num_onehot_encoded_features = 1
    cat_df = np.zeros((1, len(cat_features) - num_onehot_encoded_features + num_professions))

    for i, feature in enumerate(cat_features):
        if feature == 'Spending Score':
            cat_df[0, i] = encodings[feature][inputs[feature]]

        elif feature == 'Profession':
            profession = np.zeros(num_professions)
            profession[encodings[feature][inputs[feature]]] = 1
            cat_df[:, i:i+num_professions] = profession.reshape(1, num_professions)

        elif feature in ['Married', 'Graduated', 'Gender']:
            cat_df[0, i+num_professions-num_onehot_encoded_features] = encodings[feature][inputs[feature]]
    
    cat_df = pd.DataFrame(cat_df)
    
    predict_df = pd.concat([num_df, cat_df.add_suffix('_2')], axis=1)

    # Rename columns to match the original column names
    cols = predict_df.columns.tolist()
    new_columns = [i for i in columns]
    predict_df.columns = new_columns

    # Use the model to make a prediction
    prediction = model.predict(predict_df)
    st.write("This customer is predicted to belong to Segment", prediction[0], ": ", segment_descriptions.loc[prediction[0]])

st.markdown("For more information about the project e.g. how it was trained, the project structure, and the like, check the [Github repo](https://github.com/gersongerardcruz/customer_segmentation). The documentation and code was made to be as explanatory as possible so anyone could replicate it.")