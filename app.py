import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model


# Load the trained model
model = load_model("model.keras")

# Load the encoders and scaler
with open(file='label_encoder_gender.pkl', mode='rb') as file:
    label_encoder = pickle.load(file)

with open(file='one_hot_encoder_geography.pkl', mode='rb') as file:
    one_hot_encoder = pickle.load(file)

with open(file='standard_scaler.pkl', mode='rb') as file:
    scaler = pickle.load(file)


# Streamlit App
st.title('Customer Churn Prediction')

# User Input
geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', min_value=18, max_value=92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", min_value=0, max_value=10)
number_of_products = st.slider("Number of Products", min_value=1, max_value=4)
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Activate Member", [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One Hot Encode 'Geography'
geography_encoded = one_hot_encoder.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([
    input_data.drop('Geography', axis=1),
    geography_encoded_df
    ],
    axis=1
)

# Performing Label Encoder on 'Gender'
input_data['Gender'] = label_encoder.transform([gender])

# Scaling the data
input_data = scaler.transform(input_data)

# Prediction Churn
prediction = model.predict(input_data)
prediction_probability = prediction[0][0]

st.write(f"Churn Probability: {prediction_probability:.2f}")

if prediction_probability > 0.5:
    st.write("The Customer is likely to Churn")
else:
    st.write("The Customer is not likely to Churn")

