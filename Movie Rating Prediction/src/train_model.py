# ============================================
# MOVIE RATING PREDICTION - RANDOM FOREST
# (FIXED FOR REAL IMDB DATASET)
# ============================================

import os
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ---------- PATH HANDLING ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "movies.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "movie_rating_model.pkl")

# ---------- LOAD DATA ----------
data = pd.read_csv(DATA_PATH, encoding="latin1")

print("✅ Original Columns:")
print(data.columns)

# ---------- SELECT REQUIRED COLUMNS ----------
data = data[["Genre", "Director", "Actor 1", "Votes", "Rating"]]

# ---------- HANDLE MISSING VALUES ----------
data.dropna(inplace=True)

# Convert Votes to numeric
data["Votes"] = pd.to_numeric(data["Votes"], errors="coerce")
data.dropna(inplace=True)

# ---------- ENCODE CATEGORICAL DATA ----------
genre_encoder = LabelEncoder()
director_encoder = LabelEncoder()
actor_encoder = LabelEncoder()

data["Genre"] = genre_encoder.fit_transform(data["Genre"])
data["Director"] = director_encoder.fit_transform(data["Director"])
data["Actor 1"] = actor_encoder.fit_transform(data["Actor 1"])

# ---------- FEATURES & TARGET ----------
X = data.drop("Rating", axis=1)
y = data["Rating"]

# ---------- TRAIN MODEL ----------
model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)
model.fit(X, y)

# ---------- SAVE MODEL ----------
os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(
        (model, genre_encoder, director_encoder, actor_encoder),
        f
    )

print("\n🎉 Model trained successfully!")
print("📁 Model saved at:", MODEL_PATH)
