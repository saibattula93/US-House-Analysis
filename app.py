import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


st.title("US House Price Analysis and Prediction")

st.header("User Input")

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



# ...

# Create explanations for each feature
feature_explanations = {
    'Housing_Inventory': [
        "Low housing inventory can lead to increased demand and higher home prices.",
        "Scarce housing inventory may result in competitive bidding, driving up prices.",
        "High demand for limited homes can result in faster price appreciation."
    ],
    'Construction_Costs': [
        "Higher construction costs can lead to increased home prices.",
        "Increased material and labor costs can contribute to higher property values.",
        "Rising construction costs may lead to higher-priced new homes."
    ],
    'Land_Availability': [
        "Limited land availability can drive up home prices as land becomes more valuable.",
        "Scarce land resources can lead to higher demand for existing properties.",
        "Limited land may result in higher competition for available lots, increasing prices."
    ],
    'Interest_Rates': [
        "Lower interest rates may increase demand for homes and drive up prices.",
        "Affordable financing due to lower interest rates can lead to higher home values.",
        "Lower mortgage rates can make homeownership more accessible, boosting demand."
    ],
    'Economic_Conditions': [
        "Strong economic conditions can boost home prices as people have more purchasing power.",
        "A robust economy may result in increased job opportunities and housing demand.",
        "Positive economic growth can lead to higher consumer confidence and home buying."
    ],
    'Population_Growth': [
        "High population growth can increase demand for housing and impact prices.",
        "Increased population may lead to a housing shortage and rising prices.",
        "A growing population can drive the need for more homes, affecting prices."
    ],
    'Consumer_Confidence': [
        "High consumer confidence can lead to higher demand and prices in the housing market.",
        "Confident consumers are more likely to make long-term investments in real estate.",
        "Positive sentiment can drive people to buy homes, increasing demand."
    ],
    'Demographic_Trends': [
        "Demographic trends, such as millennial buyers, can impact housing demand.",
        "The preferences of different generations can affect the types of homes in demand.",
        "Understanding demographic shifts can help predict housing market trends."
    ],
    'House_Area_sqft': [
        "Larger houses may have higher prices due to the increased living space.",
        "More square footage can provide additional features and amenities, driving up prices.",
        "Homebuyers often pay more for larger homes with spacious layouts."
    ],
    'Location': [
        "Urban areas may have higher prices due to demand for city living.",
        "Proximity to job centers and amenities in urban areas can drive up property values.",
        "Urban properties may offer convenience and lifestyle benefits, affecting prices."
    ],
    'Amenities': [
        "Access to good schools can increase demand and property values in an area.",
        "Proximity to transportation options can make an area more desirable, impacting prices.",
        "Amenities such as parks, shopping centers, and recreational facilities can affect home values."
    ]
}

# ...

if st.button("Predict"):
    results = predict_data(user_data)
    st.subheader("Estimated Home Price in US:")
    st.info(f"$ {round(results, 2):,.2f}")

    # Randomly select and display a feature explanation
    st.header("Suppy and Demand Factors Affecting Home Prices")
    
    # Shuffle the feature explanations to display them randomly
    shuffled_features = list(feature_explanations.keys())
    random.shuffle(shuffled_features)
    
    for feature in shuffled_features:
        st.write(f"Effect of {feature.replace('_', ' ')}:")
        explanation = random.choice(feature_explanations[feature])
        st.write(f"- {explanation}")



    # # Display factors affecting home prices
    # st.header("Factors Affecting Home Prices")
    # st.write("Several factors can influence home prices, including:")
    
    # # You can provide explanations for each factor based on the user's input values
    # st.write("- Housing Inventory: Low housing inventory can lead to increased demand and higher home prices.")
    # st.write("- Construction Costs: Higher construction costs can lead to increased home prices.")
    # st.write("- Land Availability: Limited land availability can drive up home prices.")
    # st.write("- Interest Rates: Lower interest rates may increase demand for homes and drive up prices.")
    # st.write("- Economic Conditions: Strong economic conditions can boost home prices.")
    # st.write("- Population Growth: High population growth can increase demand for housing and impact prices.")
    # st.write("- Consumer Confidence: High consumer confidence can lead to higher demand and prices.")
    # st.write("- Demographic Trends: Demographic trends, such as millennial buyers, can impact housing demand.")
    # st.write("- House Area (sqft): Larger houses may have higher prices.")
    # st.write("- Location: Urban areas may have higher prices due to demand.")
    # st.write("- Amenities: Access to good schools and transportation can influence prices.")
