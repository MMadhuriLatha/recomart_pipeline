import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

class ContentBasedRecommender:
    """
    Content-Based Recommender System
    """
    def __init__(self):
        self.item_profiles = None
        self.similarity_matrix = None
        self.item_ids = None

    def fit(self, items_df: pd.DataFrame, feature_column: str):
        """
        Train the content-based recommender using item features.

        Args:
            items_df: DataFrame containing item metadata.
            feature_column: Column name containing text features for items.
        """
        # Create TF-IDF matrix for item features
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(items_df[feature_column])

        # Compute cosine similarity between items
        self.similarity_matrix = cosine_similarity(tfidf_matrix)

        # Store item IDs and profiles
        self.item_ids = items_df['item_id'].values
        self.item_profiles = items_df

    def recommend_items(self, user_history: List[int], n_recommendations: int = 10) -> pd.DataFrame:
        if self.similarity_matrix is None:
            raise ValueError("Model has not been trained. Call `fit` first.")

        # Filter user history to include only items in the similarity matrix
        valid_user_history = [item_id for item_id in user_history if item_id in self.item_ids]

        if not valid_user_history:
            print("No valid items in user history for recommendation.")
            return pd.DataFrame(columns=['item_id', 'similarity_score'])

        # Aggregate similarity scores for items in user history
        user_similarities = self.similarity_matrix[
            [self.item_ids.tolist().index(item_id) for item_id in valid_user_history]
        ].mean(axis=0)

        # Create a DataFrame of item scores
        item_scores = pd.DataFrame({
            'item_id': self.item_ids,
            'similarity_score': user_similarities
        })

        # Exclude items already in user history
        item_scores = item_scores[~item_scores['item_id'].isin(user_history)]

        # Return top-N recommendations
        return item_scores.nlargest(n_recommendations, 'similarity_score').reset_index(drop=True)   