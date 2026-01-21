import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ----------------------------
# Load model & scaler
# ----------------------------
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------
# Load dataset to get columns
# ----------------------------
df = pd.read_csv("spotify_churn_dataset.csv")

categorical_cols = ["gender", "country", "subscription_type", "device_type"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

feature_columns = df_encoded.drop("is_churned", axis=1).columns

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Spotify Churn Prediction", layout="centered")

st.title("🎵 Spotify User Churn Prediction")
st.write("Predict whether a user is likely to churn")

# ----------------------------
# User Inputs
# ----------------------------
gender = st.selectbox("Gender", df["gender"].unique())
country = st.selectbox("Country", df["country"].unique())
subscription_type = st.selectbox("Subscription Type", df["subscription_type"].unique())
device_type = st.selectbox("Device Type", df["device_type"].unique())

age = st.slider("Age", 10, 70, 25)
listening_time = st.slider("Listening Time (hrs/month)", 0, 300, 100)
songs_played_per_day = st.slider("Songs Played Per Day", 0, 150, 30)
skip_rate = st.slider("Skip Rate", 0.0, 1.0, 0.3)
ads_listened_per_week = st.slider("Ads Listened Per Week", 0, 100, 5)
offline_listening = st.selectbox("Offline Listening", [0, 1])

# ----------------------------
# Prepare input dataframe
# ----------------------------
input_data = {
    "user_id": 0,  # dummy
    "gender": gender,
    "age": age,
    "country": country,
    "subscription_type": subscription_type,
    "listening_time": listening_time,
    "songs_played_per_day": songs_played_per_day,
    "skip_rate": skip_rate,
    "device_type": device_type,
    "ads_listened_per_week": ads_listened_per_week,
    "offline_listening": offline_listening
}

input_df = pd.DataFrame([input_data])

# One-hot encode input
input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True, dtype=int)

# Align with training columns
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# Scale
input_scaled = scaler.transform(input_encoded)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ User is likely to CHURN (Probability: {probability:.2f})")
    else:
        st.success(f"✅ User is NOT likely to churn (Probability: {probability:.2f})")
