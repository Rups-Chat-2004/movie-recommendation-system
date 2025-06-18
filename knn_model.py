# knn_model.py

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    df.dropna(subset=['genres', 'title'], inplace=True)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['genres'])

    return X, vectorizer, df

def train_and_save_model(csv_file, model_path="model", model_file="knn_model.pkl"):
    X, vectorizer, df = preprocess_data(csv_file)

    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(X)

    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(model_path, model_file), "wb") as f:
        pickle.dump(knn, f)

    with open(os.path.join(model_path, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    df.to_csv(os.path.join(model_path, "movie_dataframe.csv"), index=False)

    print("KNN model and preprocessing artifacts saved.")

def load_model_and_artifacts(model_path="model", model_file="knn_model.pkl"):
    with open(os.path.join(model_path, model_file), "rb") as f:
        knn = pickle.load(f)

    with open(os.path.join(model_path, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    df = pd.read_csv(os.path.join(model_path, "movie_dataframe.csv"))

    return knn, vectorizer, df

def recommend_movies(knn, vectorizer, df, input_genre):
    input_vector = vectorizer.transform([input_genre])
    distances, indices = knn.kneighbors(input_vector)

    recommended_titles = df.iloc[indices[0]]['title'].tolist()
    return recommended_titles
