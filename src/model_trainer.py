from sklearn.neighbors import NearestNeighbors

class Movie_recommender:
    def __init__(self, metric='cosine', algorithm='brute'):
        self.model = NearestNeighbors(metric=metric, algorithm=algorithm)
    
    def fit(self, data):
        self.model.fit(data)
    
    def get_neighbors(self, movie_vector, n_neighbors=6):
        distances, indices = self.model.kneighbors(movie_vector, n_neighbors=n_neighbors)
        return distances.flatten(), indices.flatten()