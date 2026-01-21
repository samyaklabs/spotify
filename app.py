import pandas as pd
import numpy as np
import streamlit as st
import pickle

# ----------------------------
# Load model & scaler
# ----------------------------
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------
# Load dataset (only to get column structure)
# ----------------------------
df = pd.read_csv("spotify_churn_dataset.csv")

# Normalize column names (CRITICAL for Streamlit Cloud)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Categorical columns used during training
categorical_cols = ["gender", "country", "subscription_type", "device_type"]

# One-hot encode dataset to capture training feature order
df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

# Save training feature order
feature_columns = df_encoded.drop("is_churned", axis=1).columns

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Spotify Churn Prediction", layout="centered")
st.title("🎵 Music Streaming Service Churn Prediction")

# Numeric inputs
age = st.number_input("Age", 10, 80, 25)
listening_time = st.number_input("Listening Time (mins/day)", 0, 300, 60)
songs_played_per_day = st.number_input("Songs Played Per Day", 0, 150, 30)
skip_rate = st.slider("Skip Rate", 0.0, 1.0, 0.3)
ads_listened_per_week = st.number_input("Ads Listened Per Week", 0, 100, 5)
offline_listening = st.selectbox("Offline Listening", [0, 1])

# Categorical inputs
gender = st.selectbox("Gender", sorted(df["gender"].unique()))
country = st.selectbox("Country", sorted(df["country"].unique()))
subscription_type = st.selectbox(
    "Subscription Type", sorted(df["subscription_type"].unique())
)
device_type = st.selectbox("Device Type", sorted(df["device_type"].unique()))

# ----------------------------
# Create input dataframe
# ----------------------------
input_df = pd.DataFrame([{
    "user_id": 0,  # dummy value
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

# Normalize input column names
input_df.columns = input_df.columns.str.strip().str.lower().str.replace(" ", "_")

# ----------------------------
# Encode & align input with training features
# ----------------------------
input_encoded = pd.get_dummies(
    input_df,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

# Align columns with training data
input_encoded = input_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

# ----------------------------
# Scale & Predict
# ----------------------------
input_scaled = scaler.transform(input_encoded)
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# ----------------------------
# Output
# ----------------------------
if st.button("🔍 Predict Churn"):
    if prediction == 1:
        st.error(f"⚠️ User is likely to CHURN (Probability: {probability:.2f})")
    else:
        st.success(f"✅ User is NOT likely to churn (Probability: {probability:.2f})")
