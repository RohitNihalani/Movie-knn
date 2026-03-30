from src.data_loader import load_and_clean
from src.model_trainer import Movie_recommender

RATINGS_DATA=r'C:\Users\rohit\OneDrive\Desktop\movie_recommendation\data\ratings.csv'
MOVIES_DATA= r'C:\Users\rohit\OneDrive\Desktop\movie_recommendation\data\movies.csv'
def main():
    mat,sparse_mat,df_movies=load_and_clean(RATINGS_DATA,MOVIES_DATA)
    recommender=Movie_recommender()
    recommender.fit(sparse_mat)

    search_title = "Toy Story"
    movie_id = df_movies[df_movies['title'].str.contains(search_title)]['movieId'].iloc[0]
    movie_idx = mat.index.get_loc(movie_id)

    dist, idx = recommender.get_neighbors(mat.iloc[movie_idx, :].values.reshape(1, -1))

    print(f"Movies similar to {search_title}:")
    for i in range(1, len(dist)):
        actual_movie_id = mat.index[idx[i]]
        print(f"- {df_movies[df_movies['movieId'] == actual_movie_id]['title'].values[0]}")

if __name__ == "__main__":
    main()