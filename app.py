import streamlit as st
import pandas as pd
# import numpy as np
# import joblib as joblib
import requests
# from sklearn.metrics.pairwise import cosine_similarity
from recommender_model import MovieRecommender
import logging

logging.basicConfig(level=logging.INFO)

# set app config
st.set_page_config(page_title="Movies Recommendation", page_icon="🎬", layout="wide")    
st.markdown(f"""
            <style>
            .stApp {{background-image: url(""); 
                     background-attachment: fixed;
                     base: light;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)

# --- Load Recommender ---
@st.cache_resource
def load_recommender():
    logging.info("Loading recommender...")
    try:
        recommender = MovieRecommender()
        logging.info("Calculating Similarity...")
        recommender.calculate_similarity()
        logging.info("Recommender loaded successfully.")
        return recommender
    except Exception as e:
        logging.error(f"Error loading recommender: {e}")
        return None

recommender = load_recommender()
if recommender is None:
    st.error("Failed to load the movie recommender.")
    st.stop()

df_movies = recommender.df_sample
logging.info(f"Number of movies loaded: {len(df_movies)}")

# --- Fetch Poster Function ---
@st.cache_data  # Cache the poster URLs for better performance
def fetch_poster(movie_titles):
    logging.info(f"Fetching posters for: {movie_titles}")
    ids = []
    posters = []
    for title in movie_titles:
        try:
            movie_id = df_movies[df_movies['title'] == title]['id'].values[0]
            ids.append(movie_id)
        except IndexError:
            print(f"Movie ID not found for title: {title}")
            posters.append(None)  # Handle cases where ID is not found
            continue

    for movie_id in ids:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=5a2d8928f55e54df341caf236e1fc136"
        # print("\n url")
        # print(url)
        response = requests.get(url)
        # response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        poster_path = data['poster_path']
        # print("\n poster_path")
        # print(poster_path)
        if poster_path:
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            # print("\n full_path")
            # print(full_path)
            posters.append(full_path)
        else:
            posters.append(None)
    logging.info(f"Posters fetched successfully for: {movie_titles}")
    return posters

# --- Movie Recommendation and Display ---
st.title("Movie Recommendation System")

movie_titles_list = df_movies['title'].tolist()
default_movie_index = movie_titles_list.index("Toy Story") if "Toy Story" in movie_titles_list else 0

selected_movie_title = st.selectbox("Select a movie:", movie_titles_list, index=default_movie_index)

num_recommendations_options = list(range(1, 11))  # Create a list from 1 to 10
num_recommendations = st.selectbox("Number of recommendations:", num_recommendations_options, index=4) # Default to 5


if selected_movie_title:
    try:
        recommendations_df = recommender.recommend_movies(
            movie_title=selected_movie_title,
            num_recommendations=num_recommendations
        )
        recommended_movie_titles = recommendations_df['title'].tolist()
        recommended_posters = fetch_poster(recommended_movie_titles)

        st.subheader(f"Top {num_recommendations} Recommendations for '{selected_movie_title}':")

        num_movies = len(recommended_movie_titles)
        num_cols = min(5, num_movies)
        cols = st.columns(num_cols, gap='medium')
        max_poster_height = 300 

        for i in range(num_movies):
            with cols[i % num_cols]:
                with st.container():
                    st.markdown(f"<h3 style='font-size: 16px; height: 50px; overflow: hidden;'>{recommended_movie_titles[i]}</h3>", unsafe_allow_html=True)
                    if recommended_posters[i]:
                        st.image(recommended_posters[i], use_container_width=True)
                    else:
                        st.markdown("<p style='text-align: center;'>No Poster Available</p>", unsafe_allow_html=True)

        # Add CSS for container height
        st.markdown(f"""
        <style>
        .st-container {{
            height: {max_poster_height + 70}px; 
            display: flex;
            flex-direction: column;
            justify-content: flex-start; 
            align-items: stretch; 
        }}
        </style>
        """, unsafe_allow_html=True)

    except ValueError as e:
        st.error(f"Error: {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching poster: {e}")

st.markdown(f"""
<style>
body {{
    color: #333; 
}}
</style>
""", unsafe_allow_html=True)
