# ============================================
# STREAMLIT MOVIE RATING PREDICTION APP
# ============================================

import streamlit as st
import pickle

# Load model
with open("../model/movie_rating_model.pkl", "rb") as file:
    model, genre_encoder, director_encoder, actor_encoder = pickle.load(file)

st.title("🎬 Movie Rating Prediction App")
st.write("Predict movie rating using Machine Learning")

# User Inputs
genre = st.selectbox(
    "Select Genre",
    genre_encoder.classes_
)

director = st.selectbox(
    "Select Director",
    director_encoder.classes_
)

actor = st.selectbox(
    "Select Actor",
    actor_encoder.classes_
)

votes = st.number_input(
    "Number of Votes",
    min_value=1000,
    step=1000
)

# Predict Button
if st.button("Predict Rating"):
    genre_encoded = genre_encoder.transform([genre])[0]
    director_encoded = director_encoder.transform([director])[0]
    actor_encoded = actor_encoder.transform([actor])[0]

    prediction = model.predict(
        [[genre_encoded, director_encoded, actor_encoded, votes]]
    )

    st.success(f"⭐ Predicted Movie Rating: {round(prediction[0], 2)}")
