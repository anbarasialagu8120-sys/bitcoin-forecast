import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load models
model_1 = tf.keras.models.load_model("lstm_1day.keras")
model_3 = tf.keras.models.load_model("rnn_3day.keras")
model_7 = tf.keras.models.load_model("lstm_7day.keras")

# Load scaler (FIXED)
scaler = joblib.load("scaler.pkl")

st.title("📈 Bitcoin Forecast (1 / 3 / 7 Days)")

input_data = st.text_area("Enter 60 values separated by commas")

if st.button("Predict"):

    try:
        # Convert input string to float array
        data = np.array([float(x) for x in input_data.split(",")])

        # Check length
        if len(data) != 60:
            st.error("⚠️ Please enter exactly 60 values")
        else:
            # reshape for scaler
            data = data.reshape(-1, 1)

            # scale input
            data_scaled = scaler.transform(data)

            # reshape for model
            X = data_scaled.reshape(1, 60, 1)

            # predictions (scaled)
            pred1 = model_1.predict(X)
            pred3 = model_3.predict(X)
            pred7 = model_7.predict(X)

            # remove extra dimensions (IMPORTANT FIX)
            pred1 = pred1.reshape(-1)[0]
            pred3 = pred3.reshape(-1)[0]
            pred7 = pred7.reshape(-1)[0]

            # inverse transform (convert back to real price)
            pred1 = scaler.inverse_transform([[pred1]])[0][0]
            pred3 = scaler.inverse_transform([[pred3]])[0][0]
            pred7 = scaler.inverse_transform([[pred7]])[0][0]

            # output
            st.success("Predictions:")

            st.write("📅 1 Day Prediction:", pred1)
            st.write("📅 3 Day Prediction:", pred3)
            st.write("📅 7 Day Prediction:", pred7)

    except Exception as e:
        st.error(f"Error: {e}")