import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

st.title("🏠 House Price Prediction App")
st.write("Estimate the price of a house by modifying the features below.")

# Load model and a sample of training data
@st.cache_resource
def load_resources():
    model = joblib.load('model.pkl')
    # Load engineered dataset to retrieve the exact column structure
    df_train = pd.read_csv('data/engineered_data.csv')
    return model, df_train

model, df_train = load_resources()

# Interactive user widgets
st.subheader("Property Parameters")

sqft_living = st.number_input("Living area (sqft_living)", min_value=300, max_value=15000, value=2000, step=50)
bedrooms = st.slider("Number of bedrooms (bedrooms)", min_value=1, max_value=10, value=3)
bathrooms = st.slider("Number of bathrooms (bathrooms)", min_value=1.0, max_value=8.0, value=2.0, step=0.5)
grade = st.slider("Construction grade (grade)", min_value=1, max_value=15, value=7)
house_age = st.number_input("House age (house_age)", min_value=0, max_value=150, value=15)

# Prediction button
if st.button("Predict Price"):
    # 1. Duplicate a reference row to keep all columns expected by the model
    input_data = df_train.drop(columns=['price', 'date'], errors='ignore').iloc[[0]].copy()
    
    # 2. Update columns modified by the user (if they exist in the dataset)
    if 'sqft_living' in input_data.columns:
        input_data['sqft_living'] = sqft_living
    if 'bedrooms' in input_data.columns:
        input_data['bedrooms'] = bedrooms
    if 'bathrooms' in input_data.columns:
        input_data['bathrooms'] = bathrooms
    if 'grade' in input_data.columns:
        input_data['grade'] = grade
    if 'house_age' in input_data.columns:
        input_data['house_age'] = house_age

    # 3. Prediction with the model
    prediction = model.predict(input_data)[0]
    
    # 4. Display result
    st.success(f"The estimated price of this property is: **${prediction:,.2f}**")