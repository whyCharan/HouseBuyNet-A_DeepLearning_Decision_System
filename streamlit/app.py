import streamlit as st
import joblib
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_ann():
    model = Sequential([
        Dense(32, activation='relu', input_shape=(23,)), 
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


pipe = joblib.load("/mnt/data/house_buying_ann_pipeline.pkl")


st.title("🏠 House Purchase Decision Prediction")
st.write("Enter the property and customer details below to predict purchase decision.")


country_list = ['France', 'South Africa', 'Germany']

city_list = [
    'Marseille', 'Cape Town', 'Johannesburg', 'Frankfurt', 'Paris',
    'Berlin', 'Munich', 'Hamburg', 'Stuttgart', 'Lyon'
]

property_type_list = ['Farmhouse', 'Apartment', 'Townhouse']

furnishing_status_list = ['Semi-Furnished', 'Fully-Furnished', 'Unfurnished']



country = st.selectbox("Country", country_list)
city = st.selectbox("City", city_list)
property_type = st.selectbox("Property Type", property_type_list)
furnishing_status = st.selectbox("Furnishing Status", furnishing_status_list)


property_size_sqft = st.number_input("Property Size (sqft)", min_value=200, max_value=6000, step=50)
price = st.number_input("Price", min_value=10000, max_value=2000000, step=1000)
constructed_year = st.number_input("Constructed Year", min_value=1980, max_value=2025, step=1)
previous_owners = st.number_input("Previous Owners", min_value=0, max_value=7, step=1)
rooms = st.number_input("Rooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)

garage = st.selectbox("Garage", [0, 1])
garden = st.selectbox("Garden", [0, 1])

crime_cases_reported = st.number_input("Crime Cases Reported", min_value=0, max_value=3, step=1)
legal_cases_on_property = st.selectbox("Legal Cases on Property", [0, 1])
customer_salary = st.number_input("Customer Salary", min_value=1000, step=500)
loan_amount = st.number_input("Loan Amount", min_value=1000, step=1000)
loan_tenure_years = st.number_input("Loan Tenure (Years)", min_value=1, max_value=30, step=1)
monthly_expenses = st.number_input("Monthly Expenses", min_value=0, step=100)
down_payment = st.number_input("Down Payment", min_value=0, step=1000)

emi_to_income_ratio = st.number_input("EMI to Income Ratio", min_value=0.0, max_value=0.5, step=0.01)
satisfaction_score = st.number_input("Satisfaction Score (1-10)", min_value=1, max_value=10, step=1)
neighbourhood_rating = st.number_input("Neighbourhood Rating (1-10)", min_value=1, max_value=10, step=1)
connectivity_score = st.number_input("Connectivity Score (1-10)", min_value=1, max_value=10, step=1)


input_df = pd.DataFrame([[

    country,
    city,
    property_type,
    furnishing_status,
    property_size_sqft,
    price,
    constructed_year,
    previous_owners,
    rooms,
    bathrooms,
    garage,
    garden,
    crime_cases_reported,
    legal_cases_on_property,
    customer_salary,
    loan_amount,
    loan_tenure_years,
    monthly_expenses,
    down_payment,
    emi_to_income_ratio,
    satisfaction_score,
    neighbourhood_rating,
    connectivity_score

]], columns=[
    'country',
    'city',
    'property_type',
    'furnishing_status',
    'property_size_sqft',
    'price',
    'constructed_year',
    'previous_owners',
    'rooms',
    'bathrooms',
    'garage',
    'garden',
    'crime_cases_reported',
    'legal_cases_on_property',
    'customer_salary',
    'loan_amount',
    'loan_tenure_years',
    'monthly_expenses',
    'down_payment',
    'emi_to_income_ratio',
    'satisfaction_score',
    'neighbourhood_rating',
    'connectivity_score'
])


if st.button("Predict"):
    prediction = pipe.predict(input_df)[0]
    probability = pipe.predict_proba(input_df)[0] 

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.success(f"✔ The customer is likely to BUY the property. (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ The customer is NOT likely to buy. (Confidence: {(1 - probability):.2f})")
