import os
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the trained model, scaler, and encoders
with open("trained_model.pkl", 'rb') as file:
    model_rf = pickle.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

with open("label_encoder.pkl", 'rb') as file:
    label_encoders = pickle.load(file)

# Load feature names
with open("feature_names.pkl", 'rb') as file:
    feature_names = pickle.load(file)

# Load the original dataset to get unique values for dropdowns
df = pd.read_csv("netflix_dataset.csv")

# Unique values for dropdowns
countries = df["country"].unique().tolist()
ratings = df["rating"].unique().tolist()
directors = df["director"].unique().tolist()
genres = df['listed_in'].str.get_dummies(sep=', ').columns.tolist()

# Streamlit app layout
st.title("Netflix Show Type Prediction")

# Input fields for the prediction
st.header("Input Features")

# Dropdowns for inputs
selected_country = st.selectbox("Select Country", countries)
selected_rating = st.selectbox("Select Rating", ratings)
selected_director = st.selectbox("Select Director", directors)

# Genre selection as multiple choice (can select multiple genres)
selected_genres = st.multiselect("Select Genres", genres)

# Duration input
duration = st.number_input("Duration (minutes)", min_value=0, value=30)

# Prepare input data for the model
if st.button("Predict"):
    # Prepare the input data as a dictionary
    input_data = {
        "country": label_encoders["country"].transform([selected_country])[0],
        "rating": label_encoders["rating"].transform([selected_rating])[0],
        "duration": duration,
    }

    # One-hot encode the selected genres
    for genre in genres:
        input_data[genre] = 1 if genre in selected_genres else 0

    # One-hot encode the selected director
    for director in directors:
        input_data[f"director_{director}"] = 1 if director == selected_director else 0

    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    # Ensure all feature names are included, filling missing ones with 0
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
            
    # Reorder the DataFrame to match the training feature order
    input_df = input_df[feature_names]

    # Standardize the features
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model_rf.predict(input_scaled)
    
    # Decode the prediction
    prediction_label = label_encoders["type"].inverse_transform(prediction)[0]

    # Display the result
    st.success(f"The predicted type is: **{prediction_label}**")
