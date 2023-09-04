import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


st.title("US House Price Analysis and Prediction")

st.header("Customer Input")

State = st.selectbox("State", ('Pennsylvania', 'Kentucky', 'South Dakota', 'Texas', 'Tennessee',
                                'Illinois', 'Oregon', 'Wisconsin', 'Washington', 'Utah',
                                'Connecticut', 'Oklahoma', 'Maryland', 'Hawaii', 'West Virginia',
                                'Indiana', 'Maine', 'Rhode Island', 'Florida', 'Georgia',
                                'Alabama', 'Arkansas', 'Mississippi', 'New York', 'Iowa',
                                'Michigan', 'North Dakota', 'Alaska', 'Colorado', 'Virginia',
                                'Kansas', 'Ohio', 'Nebraska', 'South Carolina', 'New Hampshire',
                                'Wyoming', 'Louisiana', 'California', 'Nevada', 'Idaho',
                                'Missouri', 'Delaware', 'Massachusetts', 'New Jersey',
                                'New Mexico', 'North Carolina', 'Minnesota', 'Vermont', 'Montana',
                                'Arizona'))
Housing_Inventory = st.number_input("Housing Inventory", min_value=0,  value=12000)
Construction_Costs = st.number_input("Construction Costs", min_value=0,  value=14000)
Land_Availability = st.selectbox("Land Availability", ('Abundant', 'Limited'))
Interest_Rates = st.number_input("Interest Rates", min_value=0,  value=4)
Economic_Conditions = st.selectbox("Economic Conditions", ('Moderate', 'Weak', 'Strong'))
Population_Growth = st.number_input("Population Growth", min_value=0,  value=2)
Consumer_Confidence = st.selectbox("Consumer Confidence", ('High', 'Moderate', 'Low'))
Demographic_Trends = st.selectbox("Demographic Trends", ('Aging Population', 'Millennial Buyers'))
House_Area_sqft = st.number_input("House Area in sqft", min_value=0,  value=520)

Location = st.selectbox("Location", ('Urban', 'Suburban'))
Amenities = st.selectbox("Amenities", ('Good Schools', 'Transport'))


user_data = {
    "State": State,
    "Housing_Inventory": Housing_Inventory,
    "Construction_Costs": Construction_Costs,
    "Land_Availability": Land_Availability,
    "Interest_Rates": Interest_Rates,
    "Economic_Conditions": Economic_Conditions,
    "Population_Growth": Population_Growth,
    "Consumer_Confidence": Consumer_Confidence,
    "Demographic_Trends": Demographic_Trends,
    "House_Area_sqft": House_Area_sqft,
    "Location": Location,
    "Amenities": Amenities
}

# Function to make predictions
def predict_data(user_data):
    data = CustomData(
        State=user_data["State"],
        Housing_Inventory=user_data["Housing_Inventory"],
        Construction_Costs=user_data["Construction_Costs"],
        Land_Availability=user_data["Land_Availability"],
        Interest_Rates=user_data["Interest_Rates"],
        Economic_Conditions=user_data["Economic_Conditions"],
        Population_Growth=user_data["Population_Growth"],
        Consumer_Confidence=user_data["Consumer_Confidence"],
        Demographic_Trends=user_data["Demographic_Trends"],
        House_Area_sqft=user_data["House_Area_sqft"],
        Location=user_data["Location"],
        Amenities=user_data["Amenities"]
    )

    pred_df = data.get_data_as_data_frame()

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)

    return results[0]



if st.button("Predict"):
    results = predict_data(user_data)
    st.subheader("Estimated Home Price in US:")
    st.info(f"$ {round(results, 2):,.2f}")
