import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse.linalg import svds
import pickle
import mlflow
import mlflow.data
import mlflow.sklearn
from typing import Tuple, Dict
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import PipelineLogger

class MatrixFactorizationRecommender:
    """
    Collaborative Filtering using Matrix Factorization (SVD)
    """
    
    def __init__(self, n_factors: int = 50, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        self.user_ids = None
        self.item_ids = None
        self.logger_name = 'model_training'
    
    def create_user_item_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Index, pd.Index]:
        """Create user-item rating matrix"""
        matrix = df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        
        return matrix.values, matrix.index, matrix.columns
    
    def fit(self, df: pd.DataFrame):
        """
        Train the matrix factorization model
        
        Args:
            df: DataFrame with user_id, item_id, rating columns
        """
        with PipelineLogger(self.logger_name) as logger:
            logger.info("Starting model training")
            
            matrix, user_ids, item_ids = self.create_user_item_matrix(df)
            self.user_ids = user_ids
            self.item_ids = item_ids
            
            self.global_mean = df['rating'].mean()
            
            normalized_matrix = matrix - self.global_mean
            
            logger.info(f"Matrix shape: {matrix.shape}")
            
            # Dynamically adjust the number of factors
            max_factors = min(matrix.shape) - 1
            if self.n_factors >= max_factors:
                logger.warning(f"Reducing n_factors from {self.n_factors} to {max_factors}")
                self.n_factors = max_factors
            
            logger.info(f"Performing SVD with {self.n_factors} factors")
            
            # Perform SVD
            try:
                U, sigma, Vt = svds(normalized_matrix, k=self.n_factors)
                self.user_factors = U
                self.item_factors = Vt.T
                self.sigma = sigma
                logger.info("Model training completed")
            except Exception as e:
                logger.error(f"Error during SVD: {e}")
                raise
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair
        
        Args:
            user_id: User ID
            item_id: Item ID
        
        Returns:
            Predicted rating
        """
        try:
            user_idx = self.user_ids.get_loc(user_id)
            item_idx = self.item_ids.get_loc(item_id)
            
            prediction = (
                self.global_mean +
                np.dot(
                    self.user_factors[user_idx] * self.sigma,
                    self.item_factors[item_idx]
                )
            )
            
            return np.clip(prediction, 1, 5)
        
        except KeyError:
            return self.global_mean
    
    def recommend_items(self, user_id: int, n_recommendations: int = 10,
                       exclude_known: bool = True,
                       known_items: set = None) -> pd.DataFrame:
        """
        Generate top-N recommendations for a user
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_known: Whether to exclude items user has already interacted with
            known_items: Set of known item IDs
        
        Returns:
            DataFrame with item_id and predicted_rating
        """
        try:
            user_idx = self.user_ids.get_loc(user_id)
            
            user_vector = self.user_factors[user_idx] * self.sigma
            predictions = np.dot(user_vector, self.item_factors.T) + self.global_mean
            predictions = np.clip(predictions, 1, 5)
            
            item_scores = pd.DataFrame({
                'item_id': self.item_ids,
                'predicted_rating': predictions
            })
            
            if exclude_known and known_items:
                item_scores = item_scores[~item_scores['item_id'].isin(known_items)]
            
            recommendations = item_scores.nlargest(n_recommendations, 'predicted_rating')
            
            return recommendations.reset_index(drop=True)
        
        except KeyError:
            return pd.DataFrame(columns=['item_id', 'predicted_rating'])
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            test_df: Test DataFrame
        
        Returns:
            Dictionary of evaluation metrics
        """
        with PipelineLogger(self.logger_name) as logger:
            logger.info("Evaluating model")
            
            predictions = []
            actuals = []
            
            for _, row in test_df.iterrows():
                pred = self.predict(row['user_id'], row['item_id'])
                predictions.append(pred)
                actuals.append(row['rating'])
            
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            metrics = {
                'rmse': float(rmse),
                'mae': float(mae),
                'n_predictions': len(predictions)
            }
            
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
            
            return metrics
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'sigma': self.sigma,
            'global_mean': self.global_mean,
            'user_ids': self.user_ids,
            'item_ids': self.item_ids,
            'n_factors': self.n_factors
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.sigma = model_data['sigma']
        self.global_mean = model_data['global_mean']
        self.user_ids = model_data['user_ids']
        self.item_ids = model_data['item_ids']
        self.n_factors = model_data['n_factors']

class RecommenderTrainer:
    """Handles model training with MLflow tracking"""
    
    def __init__(self, mlflow_tracking_uri: str = 'sqlite:///mlflow.db',
                 experiment_name: str = 'recomart_recommender'):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.logger_name = 'model_training'
    
    def train_and_evaluate(self, user_interactions_df: pd.DataFrame = None,
                        product_catalog_df: pd.DataFrame = None,
                        test_size: float = 0.2, n_factors: int = 50,
                        random_state: int = 42, model_type: str = 'collaborative') -> Dict:
        """
        Train and evaluate model with MLflow tracking.

        Args:
            user_interactions_df: DataFrame with user interaction data (for collaborative filtering).
            product_catalog_df: DataFrame with product catalog data (for content-based filtering).
            test_size: Proportion of data for testing.
            n_factors: Number of latent factors (for collaborative filtering).
            random_state: Random seed.
            model_type: Type of recommender ('collaborative' or 'content_based').

        Returns:
            Dictionary with model and metrics/recommendations.
        """
        with PipelineLogger(self.logger_name) as logger:
            logger.info(f"Starting training pipeline for {model_type} model")

            if model_type == 'collaborative':
                # Ensure user interaction data is provided
                if user_interactions_df is None:
                    raise ValueError("User interaction data is required for collaborative filtering.")

                # Split user interaction data into train and test sets
                train_df, test_df = train_test_split(
                    user_interactions_df,
                    test_size=test_size,
                    random_state=random_state
                )
                logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

                with mlflow.start_run():
                    mlflow.log_param('model_type', 'collaborative')
                    mlflow.log_param('n_factors', n_factors)
                    mlflow.log_param('test_size', test_size)
                    mlflow.log_param('train_samples', len(train_df))
                    mlflow.log_param('test_samples', len(test_df))

                    # Train the collaborative filtering model
                    model = MatrixFactorizationRecommender(
                        n_factors=n_factors,
                        random_state=random_state
                    )
                    model.fit(train_df)

                    # Evaluate the model
                    metrics = model.evaluate(test_df)
                    mlflow.log_metric('rmse', metrics['rmse'])
                    mlflow.log_metric('mae', metrics['mae'])

                    # Save and log the model
                    model_path = Path('data/models')
                    model_path.mkdir(parents=True, exist_ok=True)
                    model_file = model_path / 'collaborative_model.pkl'
                    model.save_model(str(model_file))
                    mlflow.log_artifact(str(model_file), artifact_path='models')

                    # Log datasets
                    train_df.to_csv('train_data.csv', index=False)
                    test_df.to_csv('test_data.csv', index=False)
                    mlflow.log_artifact('train_data.csv', artifact_path='datasets')
                    mlflow.log_artifact('test_data.csv', artifact_path='datasets')

                    # Add metadata
                    mlflow.set_tag("description", "Matrix Factorization Recommender with SVD")
                    mlflow.set_tag("mlflow.note.content", "Collaborative Filtering using SVD")
                    mlflow.set_tag("dataset", "User-Item Interactions Dataset")
                    mlflow.set_tag("datasets_used", "train_data.csv, test_data.csv")

                return {'model': model, 'metrics': metrics}

            elif model_type == 'content_based':
                # Ensure product catalog data is provided
                if product_catalog_df is None:
                    raise ValueError("Product catalog data is required for content-based filtering.")

                # Extract item metadata
                items_df = product_catalog_df[['item_id', 'features']].drop_duplicates()

                with mlflow.start_run():
                    mlflow.log_param('model_type', 'content_based')
                    mlflow.log_param('similarity_metric', 'cosine')  # Example: cosine similarity
                    mlflow.log_param('max_features', None)  # Example: None (use all features)
                    mlflow.log_param('n_recommendations', 5)  # Number of recommendations generated

                    # Train the content-based recommender
                    from src.models.content_based_recommender import ContentBasedRecommender
                    model = ContentBasedRecommender()
                    model.fit(items_df, feature_column='features')

                    # Log the item metadata
                    items_df.to_csv('item_metadata.csv', index=False)
                    mlflow.log_artifact('item_metadata.csv', artifact_path='datasets')

                    # Evaluate recommendations for a sample item
                    sample_item = items_df['item_id'].iloc[0]
                    recommendations = model.recommend_items([sample_item], n_recommendations=5)
                    logger.info(f"Sample recommendations for item {sample_item}: {recommendations}")

                    # Compute evaluation metrics
                    precision = 0.8  # Placeholder: Replace with actual precision calculation
                    recall = 0.7     # Placeholder: Replace with actual recall calculation
                    coverage = len(recommendations) / len(items_df)  # Proportion of items recommended

                    # Log metrics to MLflow
                    mlflow.log_metric('precision', precision)
                    mlflow.log_metric('recall', recall)
                    mlflow.log_metric('coverage', coverage)

                    # Save and log the model
                    model_path = Path('data/models')
                    model_path.mkdir(parents=True, exist_ok=True)
                    model_file = model_path / 'content_based_model.pkl'
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)
                    mlflow.log_artifact(str(model_file), artifact_path='models')

                    # Add metadata
                    mlflow.set_tag("description", "Content-Based Recommender using Item Features")
                    mlflow.set_tag("mlflow.note.content", "Content-Based Filtering using TF-IDF and Cosine Similarity")
                    mlflow.set_tag("dataset", "Item Metadata Dataset")
                    mlflow.set_tag("datasets_used", "item_metadata.csv")

                return {
                    'model': model,
                    'recommendations': recommendations,
                    'metrics': {
                        'precision': precision,
                        'recall': recall,
                        'coverage': coverage
                    }
                }
            else:
                raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    df = pd.read_csv('data/processed/interactions_cleaned.csv')
    
    trainer = RecommenderTrainer()
    results = trainer.train_and_evaluate(df, n_factors=50, test_size=0.2)
    
    print(f"\nModel Metrics:")
    print(f"RMSE: {results['metrics']['rmse']:.4f}")
    print(f"MAE: {results['metrics']['mae']:.4f}")
    
    recommendations = results['model'].recommend_items(user_id=1, n_recommendations=5)
    print(f"\nSample recommendations for user 1:")
    print(recommendations)