import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging

logging.basicConfig(level=logging.INFO)

# OOP movie recommendation model
class MovieRecommender:
    """
    A class for generating movie recommendations based on multiple attributes
    using TF-IDF and cosine similarity.
    """

    def __init__(self, data_path="sample_10000_for_recommender_model_df.pkl", similarity_dir="models/"):
        """
        Initializes the MovieRecommender with the movie data path.
        """
        self.data_path = data_path
        self.df_sample = pd.read_pickle(self.data_path)
        self.similarity_matrices = {}
        self.weights = {}
        self.recommended_movies = None
        logging.info(f"Initializing MovieRecommender with data path: {self.data_path}")
        self.similarity_dir = similarity_dir
        self._load_similarity_matrices()


    def calculate_similarity(self, attributes=('title', 'genres', 'overview', 'cast', 'director', 'producer')):
        """
        Calculates cosine similarity matrices for the specified movie attributes.

        Args:
            attributes (tuple, optional): The movie attributes to calculate similarity for.
                Defaults to ('title', 'genres', 'overview', 'cast', 'director', 'producer').
        """
        logging.info("Calculating similarity matrices...")
        tfidf = TfidfVectorizer(stop_words='english')
        try:
            for attr in attributes:
                logging.info(f"Processing attribute: {attr}")
                tfidf_matrix = tfidf.fit_transform(self.df_sample[attr].fillna(''))
                self.similarity_matrices[attr] = cosine_similarity(tfidf_matrix, tfidf_matrix)
            logging.info("Similarity matrices calculated successfully.")
        except Exception as e:
            logging.error(f"Error calculating similarity: {e}")
            raise Exception(f"Error calculating similarity: {e}")
        
    
    def _load_similarity_matrices(self):
        """
        Loads pre-calculated similarity matrices from the specified directory.
        """
        logging.info(f"Loading saved similarity matrices from: {self.similarity_dir}")
        expected_files = {
            'title': 'cosine_similarity_title.joblib',
            'genres': 'cosine_similarity_genres.joblib',
            'overview': 'cosine_similarity_overview.joblib',
            'cast': 'cosine_similarity_cast.joblib',
            'director': 'cosine_similarity_director.joblib',
            'producer': 'cosine_similarity_producer.joblib'
        }
        for attr, filename in expected_files.items():
            filepath = self.similarity_dir + filename
            try:
                self.similarity_matrices[attr] = joblib.load(filepath)
                logging.info(f"Loaded similarity matrix for {attr} from {filepath}")
            except FileNotFoundError:
                logging.warning(f"Saved similarity matrix not found for {attr} at {filepath}. Will calculate and save.")
                return 
        logging.info("Finished loading saved similarity matrices.")
        

    def get_recommendations(self, movie_index, num_recommendations=10):
        """
        Generates movie recommendations for a given movie index based on weighted similarity scores.

        Args:
            movie_index (int): The index of the movie to get recommendations for.
            num_recommendations (int, optional): The number of recommendations to return. Defaults to 10.

        Returns:
            pandas.DataFrame: A DataFrame containing the recommended movies, sorted by combined similarity score.
        """
        if not self.weights:
            raise ValueError("Weights must be set before getting recommendations.")
        if not self.similarity_matrices:
            raise ValueError("Similarity matrices must be calculated before getting recommendations.")

        # Calculate weighted similarity scores
        weighted_scores = np.zeros(len(self.df_sample))
        for attr, similarity_matrix in self.similarity_matrices.items():
            weighted_scores += self.weights[attr] * similarity_matrix[movie_index]

        # Get the indices of the top recommendations, excluding the input movie itself
        similar_indices = np.argsort(weighted_scores)[::-1][1:num_recommendations + 1]

        # Return the recommended movies as a DataFrame
        recommended_movies = self.df_sample.iloc[similar_indices].copy() # Ensure no changes to original
        recommended_movies['combined_similarity'] = weighted_scores[similar_indices] #add similarity score
        return recommended_movies

    def set_weights(self, weights):
        """
        Sets the weights for each attribute used in the recommendation process.

        Args:
            weights (dict): A dictionary mapping attribute names to their corresponding weights.
        """
        # Basic input validation
        if not isinstance(weights, dict):
            raise TypeError("Weights must be a dictionary.")
        if set(weights.keys()) != set(self.similarity_matrices.keys()):
            raise ValueError("Weights must be provided for all attributes used in similarity calculation.")
        if not all(isinstance(value, (int, float)) for value in weights.values()):
            raise TypeError("All weights must be numeric.")
        if not np.isclose(sum(weights.values()), 1):
            raise ValueError("Weights must sum to 1.")
        self.weights = weights

    def find_optimal_weights(self, target_movie_index, num_iterations=1000):
        """
        Finds a set of weights that maximizes the average similarity of the top recommendations
        for a given movie, using a randomized approach.

        Args:
            target_movie_index (int): The index of the target movie.
            num_iterations (int, optional): The number of random weight sets to try. Defaults to 1000.

        Returns:
            dict: The optimal set of weights found.
        """
        best_weights = None
        best_avg_similarity = -1  # Initialize with a very low value

        for _ in range(num_iterations):
            # Generate a random set of weights using a Dirichlet distribution
            weights = dict(zip(self.similarity_matrices.keys(), np.random.dirichlet(np.ones(len(self.similarity_matrices)))))

            self.set_weights(weights)  # Set the weights
            recommendations_df = self.get_recommendations(target_movie_index)
            avg_similarity = recommendations_df['combined_similarity'].mean()

            if avg_similarity > best_avg_similarity:
                best_avg_similarity = avg_similarity
                best_weights = weights

        return best_weights

    def recommend_movies(self, movie_title=None, num_recommendations=10):
        """
        Recommends movies based on a given movie title.  It uses the
        find_optimal_weights method to determine the best weights.

        Args:
            movie_title (str): The title of the movie to find recommendations for.
            num_recommendations (int): number of movies to recommend

        Returns:
           pd.DataFrame: A DataFrame containing the recommended movies

        """
        # self.calculate_similarity()
        if not self.similarity_matrices:
            logging.info("Similarity matrices not loaded. Calculating and saving...")
            #self.calculate_similarity() #removed
            raise ValueError("Similarity matrices not loaded.  Please pre-calculate and save them.")
        
        if movie_title:
            try:
                target_movie_index = self.df_sample[self.df_sample['title'] == movie_title].index[0]
                print(f"Recommendations for '{movie_title}':")
            except IndexError:
                raise ValueError(f"Movie title '{movie_title}' not found in the dataset.")
        else:
            target_movie_index = self.df_sample.sample(n=1).index.values[0]
            print(f"Recommendations for '{self.df_sample.iloc[target_movie_index].title}':")
        
        optimal_weights = self.find_optimal_weights(target_movie_index)
        print("\n Best Optimal Weights: ")
        print(optimal_weights)
        self.set_weights(optimal_weights)
        recommendations = self.get_recommendations(target_movie_index, num_recommendations)
        print("\n Best Recommendations: ")
        print(recommendations)
        return recommendations