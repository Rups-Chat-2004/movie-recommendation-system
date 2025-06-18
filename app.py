import streamlit as st
from PIL import Image
from knn_model import load_model_and_artifacts, recommend_movies

# Load your KNN model and data only once
knn, vectorizer, df = load_model_and_artifacts()

# Set Streamlit page config
st.set_page_config(page_title="CineOdysseus", layout="wide")

# Sidebar navigation
page = st.sidebar.radio("Navigate", [" Home", "Recommend"])

# ------------------------
# PAGE 1: IMAGE LANDING
# ------------------------
if page == " Home":
    st.markdown("<h1 style='text-align: center;'>Welcome to CineOdysseus</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Find your next favorite movie in seconds</h4>", unsafe_allow_html=True)
    
    # Centered image
    image = Image.open("cineodysseus_banner.png")
    st.image(image, use_container_width=True)


# ------------------------
# PAGE 2: RECOMMENDATIONS
# ------------------------
elif page == "Recommend":
    st.title(" CineOdysseus Movie Recommender")

    st.markdown("Enter a genre (e.g., **Action**, **Comedy**, **Romance**) to get recommendations.")

    # Input box
    genre_input = st.text_input("Enter Genre")

    # Recommend button
    if st.button("Recommend") and genre_input:
        try:
            results = recommend_movies(knn, vectorizer, df, genre_input)
            st.subheader("Top Recommendations:")
            for movie in results:
                st.markdown(f"{movie}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
