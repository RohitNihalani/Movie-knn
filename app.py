import streamlit as st
import pandas as pd
from src.data_loader import load_and_clean
from src.model_trainer import Movie_recommender
import os 

st.set_page_config(page_title="Movie Matcher KNN", layout="wide")

@st.cache_resource 
def initialize_engine():
    base_path = os.path.dirname(__file__)

    ratings_path = os.path.join(base_path, 'data', 'ratings.csv')
    movies_path = os.path.join(base_path, 'data', 'movies.csv')

    mat, sparse_mat, df_movies = load_and_clean(ratings_path, movies_path)
    recommender = Movie_recommender()
    recommender.fit(sparse_mat)
    return mat, df_movies, recommender

# --- UI Layout ---
st.title("🎬 Movie Recommender System")
st.markdown("Using **K-Nearest Neighbors** to find films similar to your favorites.")

try:
  
    mat, df_movies, recommender = initialize_engine()

    movie_list = df_movies['title'].values
    selected_movie = st.selectbox("Type or select a movie you like:", movie_list)

    if st.button('Show Recommendations'):
        movie_id = df_movies[df_movies['title'] == selected_movie]['movieId'].iloc[0]
        movie_idx = mat.index.get_loc(movie_id)
        
        distances, indices = recommender.get_neighbors(
            mat.iloc[movie_idx, :].values.reshape(1, -1), 
            n_neighbors=11 
        )
        
        st.subheader(f"If you liked '{selected_movie}', you might also enjoy:")
        
      
        cols = st.columns(2)
        for i in range(1, len(distances)):
            actual_movie_id = mat.index[indices[i]]
            title = df_movies[df_movies['movieId'] == actual_movie_id]['title'].values[0]
            score = round((1 - distances[i]) * 100, 2) 
            
            with cols[i % 2]:
                st.info(f"**{title}** \n*Similarity Score: {score}%*")

except Exception as e:
    st.error("Please ensure your 'data/raw/' folder contains 'movies.csv' and 'ratings.csv'.")
    st.write(e)
