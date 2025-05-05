import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load model and scaler
model = joblib.load('./mlmodel.pk')
scaler = joblib.load("Scaler.pkl")

# Streamlit page config
st.set_page_config(layout="wide")
st.title("Restaurant Rating Prediction App")
st.caption("This app helps you to predict a restaurant's review class")
st.divider()

# Inputs
averagecost = st.number_input("Please enter the estimated average cost for two", min_value=50, max_value=999999, value=1000, step=200)
tablebooking = st.selectbox("Restaurant has table booking?", ["Yes", "No"])
onlinedelivery = st.selectbox("Restaurant has online delivery?", ["Yes", "No"])
pricerange = st.selectbox("What is the price range (1 = cheapest, 4 = most expensive)", [1, 2, 3, 4])

predictbutton = st.button("Predict")

# On click
if predictbutton:
    # Encode inputs
    bookingstatus = 1 if tablebooking == "Yes" else 0
    deliverystatus = 1 if onlinedelivery == "Yes" else 0

    # Combine all features (before scaling)
    input_features = np.array([[averagecost, bookingstatus, deliverystatus, pricerange]])

    # Scale all features
    scaled_features = scaler.transform(input_features)

    # Predict
    prediction = model.predict(scaled_features)
    predicted_rating = prediction[0]
    st.write("Predicted Rating Class:", round(predicted_rating, 2))

    # Interpret prediction
    if predicted_rating < 2.5:
        st.write("Rating Interpretation: **Poor**")
    elif predicted_rating < 3.5:
        st.write("Rating Interpretation: **Average**")
    elif predicted_rating < 4.5:
        st.write("Rating Interpretation: **Good**")
    elif predicted_rating < 4.8:
        st.write("Rating Interpretation: **Very Good**")
    else:
        st.write("Rating Interpretation: **Excellent**")