import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('mlmodel.pk')     # Must be trained on 6 features
scaler = joblib.load('Scaler.pkl')    # Must be fit on 6 features

# Streamlit app settings
st.set_page_config(page_title="🍽️ Restaurant Rating Predictor", layout="wide")
st.title("🍽️ Restaurant Rating Prediction App")
st.caption("Predict a restaurant's customer rating class based on features.")
st.divider()

# Layout using columns
col1, col2 = st.columns(2)

with col1:
    averagecost = st.number_input("Estimated Average Cost for Two (₹)", min_value=50, max_value=100000, value=1000, step=100)
    pricerange = st.selectbox("Price Range (1 = Cheapest, 4 = Most Expensive)", [1, 2, 3, 4])
    votes = st.number_input("Number of Reviews (Votes)", min_value=0, max_value=50000, value=100, step=10)

with col2:
    tablebooking = st.selectbox("Table Booking Available?", ["Yes", "No"])
    onlinedelivery = st.selectbox("Online Delivery Available?", ["Yes", "No"])
    cuisines = st.multiselect("Cuisines Offered", [
        "North Indian", "South Indian", "Chinese", "Fast Food", "Italian",
        "Continental", "Cafe", "Desserts", "Bakery", "Mexican"
    ])

# Predict button
predictbutton = st.button("🔍 Predict Restaurant Rating")

# On click
if predictbutton:
    try:
        # Encode categorical features
        bookingstatus = 1 if tablebooking == "Yes" else 0
        deliverystatus = 1 if onlinedelivery == "Yes" else 0
        num_cuisines = len(cuisines)

        # Prepare input
        input_data = np.array([[averagecost, bookingstatus, deliverystatus, pricerange, votes, num_cuisines]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        predicted_rating = prediction[0]

        # Display result
        st.subheader(f"⭐ Predicted Rating Class: {round(predicted_rating, 2)}")

        # Rating interpretation
        if predicted_rating < 2.5:
            st.error("Rating Interpretation: **Poor** 😞")
        elif predicted_rating < 3.5:
            st.warning("Rating Interpretation: **Average** 😐")
        elif predicted_rating < 4.5:
            st.info("Rating Interpretation: **Good** 🙂")
        elif predicted_rating < 4.8:
            st.success("Rating Interpretation: **Very Good** 😃")
        else:
            st.balloons()
            st.success("Rating Interpretation: **Excellent** 🤩")

        # Model confidence (optional)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled_input)
            confidence = np.max(proba) * 100
            st.write(f"🧠 Model Confidence: **{confidence:.2f}%**")

        # Download result
        result_df = pd.DataFrame({
            "Average Cost": [averagecost],
            "Table Booking": [tablebooking],
            "Online Delivery": [onlinedelivery],
            "Price Range": [pricerange],
            "Votes": [votes],
            "No. of Cuisines": [num_cuisines],
            "Predicted Rating": [round(predicted_rating, 2)]
        })

        st.download_button(
            label="📥 Download Result",
            data=result_df.to_csv(index=False),
            file_name="rating_prediction.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Something went wrong during prediction: {str(e)}")
