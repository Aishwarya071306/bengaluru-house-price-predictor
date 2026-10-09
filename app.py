import streamlit as st
import json
import numpy as np
import xgboost as xgb

# 1. Page Configuration & Title
st.set_page_config(page_title="Bengaluru House Price Predictor", page_icon="🏠")
st.title("Bengaluru House Price Prediction Engine 🏠")

# 2. Load model and columns using native XGBoost JSON
@st.cache_resource
def load_saved_model():
    model = xgb.XGBRegressor()
    model.load_model("model.json")  # Reads native JSON format safely
    return model

@st.cache_data
def load_saved_columns():
    with open("columns.json", "r") as f:
        return json.load(f)['data_columns']

# Load artifacts
model = load_saved_model()
data_columns = load_saved_columns()

# Extract location options (skip first 4 numeric features: sqft, bath, balcony, bhk)
location_options = [col.title() for col in data_columns[4:]]

# 3. Prediction Helper Function
def predict_price(location, sqft, bath, bhk, balcony):
    loc_input = location.lower().strip()
    loc_index = data_columns.index(loc_input) if loc_input in data_columns else -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = balcony
    x[3] = bhk
    
    if loc_index >= 0:
        x[loc_index] = 1

    # Predict log scale price -> convert back with expm1
    pred_log = model.predict(np.array([x]))[0]
    return np.expm1(pred_log)

# 4. User Interface Elements
st.write("Enter the property details below to estimate the price in Lakhs (INR):")

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Select Location", sorted(location_options))
    sqft = st.number_input("Total Square Feet Area", min_value=300, max_value=10000, value=1200, step=50)

with col2:
    bhk = st.selectbox("Bedrooms (BHK)", [1, 2, 3, 4, 5], index=1)
    bath = st.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1)
    balcony = st.selectbox("Balconies", [0, 1, 2, 3], index=1)

# 5. Prediction Action
if st.button("Predict Price"):
    estimated_price = predict_price(location, sqft, bath, bhk, balcony)
    st.success(f"Estimated Property Price: **₹ {estimated_price:.2f} Lakhs**")