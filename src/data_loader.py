import pandas as pd

from scipy.sparse import csr_matrix

def load_and_clean(ratings_path, movies_path):
    df_ratings = pd.read_csv(ratings_path)
    
    
    movie_user_mat = df_ratings.pivot(
        index='movieId', columns='userId', values='rating'
    ).fillna(0)
    df_movies = pd.read_csv(movies_path)

    sparse_data = csr_matrix(movie_user_mat.values)
    
    return movie_user_mat, sparse_data, df_movies