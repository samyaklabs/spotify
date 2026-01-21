import pandas as pd
import numpy as np
import streamlit as st
import pickle

# -----------------------
# Load model & scaler
# -----------------------
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------
# Load dataset to get columns
# -----------------------
df = pd.read_csv("spotify_churn_dataset.csv")

categorical_cols = ["gender", "country", "subscription_type", "device_type"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
feature_columns = df_encoded.drop("is_churned", axis=1).columns

# -----------------------
# UI
# -----------------------
st.title("Music Streaming Service Churn Prediction")

age = st.number_input("Age", 10, 80, 25)
listening_time = st.number_input("Listening Time", 0, 300, 60)
songs_played_per_day = st.number_input("Songs Played Per Day", 0, 150, 30)
skip_rate = st.slider("Skip Rate", 0.0, 1.0, 0.3)
ads_listened_per_week = st.number_input("Ads per Week", 0, 100, 5)
offline_listening = st.selectbox("Offline Listening", [0, 1])

gender = st.selectbox("Gender", df["gender"].unique())
country = st.selectbox("Country", df["country"].unique())
subscription_type = st.selectbox("Subscription Type", df["subscription_type"].unique())
device_type = st.selectbox("Device Type", df["device_type"].unique())

# -----------------------
# Create input dataframe
# -----------------------
input_df = pd.DataFrame([{
    "user_id": 0,
    "age": age,
    "listening_time": listening_time,
    "songs_played_per_day": songs_played_per_day,
    "skip_rate": skip_rate,
    "ads_listened_per_week": ads_listened_per_week,
    "offline_listening": offline_listening,
    "gender": gender,
    "country": country,
    "subscription_type": subscription_type,
    "device_type": device_type
}])

# -----------------------
# Encode & align
# -----------------------
input_encoded = pd.get_dummies(
    input_df,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

input_encoded = input_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

# -----------------------
# Scale & Predict
# -----------------------
input_scaled = scaler.transform(input_encoded)
prediction = model.predict(input_scaled)[0]
prob = model.predict_proba(input_scaled)[0][1]

# -----------------------
# Output
# -----------------------
if prediction == 1:
    st.error(f"⚠️ Likely to CHURN (Probability: {prob:.2f})")
else:
    st.success(f"✅ Not likely to churn (Probability: {prob:.2f})")
