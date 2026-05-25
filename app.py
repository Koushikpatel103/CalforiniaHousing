import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("California Housing Price Prediction")

st.write(
    "Enter housing details below"
)

# Inputs
longitude = st.number_input(
    "Longitude"
)

latitude = st.number_input(
    "Latitude"
)

housing_median_age = st.number_input(
    "Housing Median Age"
)

total_rooms = st.number_input(
    "Total Rooms"
)

total_bedrooms = st.number_input(
    "Total Bedrooms"
)

population = st.number_input(
    "Population"
)

households = st.number_input(
    "Households"
)

median_income = st.number_input(
    "Median Income"
)

ocean = st.selectbox(
    "Ocean Proximity",
    [
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]
)

# Predict button
if st.button("Predict Price"):

    rooms_per_household = (
        total_rooms /
        households
        if households != 0 else 0
    )

    bedrooms_per_room = (
        total_bedrooms /
        total_rooms
        if total_rooms != 0 else 0
    )

    population_per_household = (
        population /
        households
        if households != 0 else 0
    )

    data = {
        'longitude':[longitude],
        'latitude':[latitude],
        'housing_median_age':[housing_median_age],
        'total_rooms':[total_rooms],
        'total_bedrooms':[total_bedrooms],
        'population':[population],
        'households':[households],
        'median_income':[median_income],
        'rooms_per_household':[rooms_per_household],
        'bedrooms_per_room':[bedrooms_per_room],
        'population_per_household':[
            population_per_household
        ]
    }

    input_df = pd.DataFrame(data)

    # Encoding
    input_df['ocean_proximity_INLAND'] = (
        1 if ocean == "INLAND"
        else 0
    )

    input_df['ocean_proximity_ISLAND'] = (
        1 if ocean == "ISLAND"
        else 0
    )

    input_df['ocean_proximity_NEAR BAY'] = (
        1 if ocean == "NEAR BAY"
        else 0
    )

    input_df['ocean_proximity_NEAR OCEAN'] = (
        1 if ocean == "NEAR OCEAN"
        else 0
    )

    # Match columns
    input_df = input_df.reindex(
        columns=columns,
        fill_value=0
    )

    # Scale
    input_scaled = scaler.transform(
        input_df
    )

    prediction = model.predict(
        input_scaled
    )

    st.success(
        f"Predicted House Price: ${prediction[0]:,.2f}"
    )
    st.subheader("Model Performance")

    st.write("R² Score: 0.63")
    st.write("MAE: $50,492")
    