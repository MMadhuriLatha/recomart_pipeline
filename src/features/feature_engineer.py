import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import PipelineLogger
from src.utils.storage import DataLakeStorage

class FeatureEngineer:
    """Creates features for recommendation systems"""
    
    def __init__(self, storage: DataLakeStorage = None):
        self.storage = storage or DataLakeStorage()
        self.logger_name = 'feature_engineering'
    
    def create_user_features(self, df: pd.DataFrame, logger=None) -> pd.DataFrame:
        """
        Create user-level aggregated features
        
        Features:
        - Total interactions count
        - Average rating given
        - Rating variance/std
        - Interaction recency
        """
        if logger:
            logger.info("Creating user features")
            
        user_features = df.groupby('user_id').agg({
            'item_id': 'count',
            'rating': ['mean', 'std', 'min', 'max'],
            'timestamp': ['min', 'max']
        }).reset_index()
        
        user_features.columns = [
            'user_id', 
            'interaction_count',
            'avg_rating',
            'rating_std',
            'min_rating',
            'max_rating',
            'first_interaction',
            'last_interaction'
        ]
        
        user_features['rating_std'] = user_features['rating_std'].fillna(0)
        
        user_features['rating_range'] = (
            user_features['max_rating'] - user_features['min_rating']
        )
                    
        if 'timestamp' in df.columns:
            latest_date = pd.to_datetime(df['timestamp']).max()
            user_features['last_interaction'] = pd.to_datetime(user_features['last_interaction'])
            user_features['days_since_last_interaction'] = (
                (latest_date - user_features['last_interaction']).dt.days
            )
            
        user_features['interaction_frequency'] = (
            user_features['interaction_count'] / user_features['days_since_last_interaction'].replace(0, 1))

        
        logger.info(f"Created {len(user_features.columns)} user features for {len(user_features)} users")
        
        return user_features
    
    def create_item_features(self, df: pd.DataFrame, product_df: pd.DataFrame = None, logger=None) -> pd.DataFrame:
        """
        Create item-level aggregated features
        
        Features:
        - Popularity (interaction count)
        - Average rating received
        - Rating variance
        - Product metadata features
        """
        if logger:
            logger.info("Creating item features")
            
        item_features = df.groupby('item_id').agg({
            'user_id': 'count',
            'rating': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        item_features.columns = [
            'item_id',
            'popularity_score',
            'avg_item_rating',
            'item_rating_std',
            'min_item_rating',
            'max_item_rating'
        ]
        
        item_features['item_rating_std'] = item_features['item_rating_std'].fillna(0)
        
        max_popularity = item_features['popularity_score'].max()
        item_features['popularity_normalized'] = (
            item_features['popularity_score'] / max_popularity
        )
        
        item_features['rating_consistency'] = (
            1 - (item_features['item_rating_std'] / 4)
        ).clip(0, 1)
        
        if product_df is not None:
            item_features = item_features.merge(
                product_df[['item_id', 'price', 'category']],
                left_on='item_id',
                right_on='item_id',
                how='left'
            )
        
        logger.info(f"Created {len(item_features.columns)} item features for {len(item_features)} items")
        
        return item_features
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                   user_features: pd.DataFrame,
                                   item_features: pd.DataFrame, logger=None) -> pd.DataFrame:
        """
        Create interaction-level features by combining user and item features
        """
        if logger:
            logger.info("Creating interaction features")
            
        df = df.merge(user_features, on='user_id', how='left')
        df = df.merge(item_features, on='item_id', how='left')
        
        df['user_item_rating_diff'] = df['rating'] - df['avg_item_rating']
        
        df['rating_deviation_from_user_avg'] = df['rating'] - df['avg_rating']
        
        df['user_popularity_affinity'] = df['interaction_count'] * df['popularity_normalized']
        
        logger.info(f"Created interaction features. Total columns: {len(df.columns)}")
        
        return df
    
    def create_similarity_features(self, df: pd.DataFrame, top_k: int = 10, logger=None) -> pd.DataFrame:
        """
        Create item-item similarity features using cosine similarity
        """
        if logger:
            logger.info("Creating similarity features")
            
        user_item_matrix = df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        
        sparse_matrix = csr_matrix(user_item_matrix.values)
        item_similarity = cosine_similarity(sparse_matrix.T)
        
        similarity_df = pd.DataFrame(
            item_similarity,
            index=user_item_matrix.columns,
            columns=user_item_matrix.columns
        )
        
        logger.info(f"Computed similarity matrix: {similarity_df.shape}")
        
        return similarity_df
    
    def create_temporal_features(self, df: pd.DataFrame, logger=None) -> pd.DataFrame:
        """Create time-based features"""
        if logger:
            logger.info("Creating temporal features")
            
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.sort_values(['user_id', 'timestamp'])
            
            df['interaction_number'] = df.groupby('user_id').cumcount() + 1
            
            df['time_since_first'] = (
                df.groupby('user_id')['timestamp']
                .transform(lambda x: (x - x.min()).dt.total_seconds() / 3600)
            )
            
            df['time_diff'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
            df['time_diff'] = df['time_diff'].fillna(0)
        
        logger.info("Temporal features created")
        
        return df
    
    def engineer_all_features(self, interactions_df: pd.DataFrame, 
                            products_df: pd.DataFrame = None, logger=None) -> dict:
        """
        Execute complete feature engineering pipeline
        
        Returns:
            Dictionary containing all feature datasets
        """
        with PipelineLogger(self.logger_name) as logger:
            logger.info("Starting complete feature engineering pipeline")
            
            # Step 1: Create user features
            user_features = self.create_user_features(interactions_df, logger=logger)
            logger.info(f"User features created: {user_features.shape}")
            
            # Step 2: Create item features
            item_features = self.create_item_features(interactions_df, products_df, logger=logger)
            logger.info(f"Item features created: {item_features.shape}")
            
            # Step 3: Create interaction features
            interactions_with_features = self.create_interaction_features(
                interactions_df,
                user_features,
                item_features, 
                logger=logger
            )
            logger.info(f"Interaction features created: {interactions_with_features.shape}")
            
            # Step 4: Create temporal features
            interactions_with_features = self.create_temporal_features(interactions_with_features, logger=logger)
            logger.info(f"Temporal features added: {interactions_with_features.shape}")


            # Step 5: Create similarity-based features
            similarity_features = self.create_similarity_features(interactions_df, logger=logger)
            logger.info(f"Similarity features created: {similarity_features.shape}")
            
            # Step 6: Save features to the feature store
            self.storage.save_features(user_features, 'user_features', 'v1')
            self.storage.save_features(item_features, 'item_features', 'v1')
            self.storage.save_features(interactions_with_features, 'interaction_features', 'v1')
            self.storage.save_features(similarity_features, 'similarity_features', 'v1')
            logger.info("All features saved to the feature store")
                
            logger.info("Feature engineering pipeline completed successfully")
            
            return {
                'user_features': user_features,
                'item_features': item_features,
                'interaction_features': interactions_with_features,
                'similarity_features': similarity_features 
            }

if __name__ == "__main__":
    engineer = FeatureEngineer()
    
    interactions_df = pd.read_csv('data/processed/interactions_cleaned.csv')
    
    features = engineer.engineer_all_features(interactions_df)
    
    print(f"\nUser features: {features['user_features'].shape}")
    print(f"Item features: {features['item_features'].shape}")
    print(f"Interaction features: {features['interaction_features'].shape}")
    print(f"Similarity features: {features['similarity_features'].shape}")
