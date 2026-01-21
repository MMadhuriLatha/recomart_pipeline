import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import PipelineLogger
from src.utils.storage import DataLakeStorage

class DataCleaner:
    """Handles data cleaning and preprocessing"""
    
    def __init__(self, storage: DataLakeStorage = None):
        self.storage = storage or DataLakeStorage()
        self.logger_name = 'data_preparation'
        self.label_encoders = {}
        self.scalers = {}
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: dict = None) -> pd.DataFrame:
        """
        Handle missing values based on strategy
        
        Args:
            df: Input DataFrame
            strategy: Dict mapping column names to strategies ('drop', 'mean', 'median', 'mode', value)
        
        Returns:
            DataFrame with missing values handled
        """
        with PipelineLogger(self.logger_name) as logger:
            df = df.copy()
            
            if strategy is None:
                strategy = {}
                
            exclude_columns = ['event_type','extra_column']
            if exclude_columns:
                columns_to_check = [col for col in df.columns if col not in exclude_columns]
            else:
                columns_to_check = df.columns
            
            for col in columns_to_check:
                if df[col].isnull().any():
                    strat = strategy.get(col, 'drop')
                    
                    if strat == 'drop':
                        before = len(df)
                        df = df.dropna(subset=[col])
                        logger.info(f"Dropped {before - len(df)} rows with missing {col}")
                    
                    elif strat == 'mean' and df[col].dtype in ['int64', 'float64']:
                        fill_value = df[col].mean()
                        df[col].fillna(fill_value, inplace=True)
                        logger.info(f"Filled {col} with mean: {fill_value:.2f}")
                    
                    elif strat == 'median' and df[col].dtype in ['int64', 'float64']:
                        fill_value = df[col].median()
                        df[col].fillna(fill_value, inplace=True)
                        logger.info(f"Filled {col} with median: {fill_value:.2f}")
                    
                    elif strat == 'mode':
                        fill_value = df[col].mode()[0]
                        df[col].fillna(fill_value, inplace=True)
                        logger.info(f"Filled {col} with mode: {fill_value}")
                    
                    else:
                        df[col].fillna(strat, inplace=True)
                        logger.info(f"Filled {col} with value: {strat}")
            
            return df
    
    def remove_duplicates(self, df: pd.DataFrame, subset: list = None, keep: str = 'last') -> pd.DataFrame:
        """Remove duplicate records"""
        with PipelineLogger(self.logger_name) as logger:
            before = len(df)
            df = df.drop_duplicates(subset=subset, keep=keep)
            removed = before - len(df)
            logger.info(f"Removed {removed} duplicate records")
            return df
    
    def encode_categorical(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Encode categorical variables"""
        with PipelineLogger(self.logger_name) as logger:
            df = df.copy()
            
            for col in columns:
                if col in df.columns:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
                    else:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                    
                    logger.info(f"Encoded {col}: {df[col].nunique()} unique values")
            
            return df
    
    def normalize_numerical(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Normalize numerical columns to [0, 1] range"""
        with PipelineLogger(self.logger_name) as logger:
            df = df.copy()
            
            for col in columns:
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    if col not in self.scalers:
                        self.scalers[col] = MinMaxScaler()
                        df[f'{col}_normalized'] = self.scalers[col].fit_transform(df[[col]])
                    else:
                        df[f'{col}_normalized'] = self.scalers[col].transform(df[[col]])
                    
                    logger.info(f"Normalized {col}: range [{df[col].min():.2f}, {df[col].max():.2f}]")
            
            return df
    
    def process_timestamps(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Extract features from timestamp column."""
        with PipelineLogger(self.logger_name) as logger:
            df = df.copy()
            
            if timestamp_col in df.columns:
                try:
                    # Automatically infer the format of the timestamp
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='mixed')
                    
                    # Extract temporal features
                    df['year'] = df[timestamp_col].dt.year
                    df['month'] = df[timestamp_col].dt.month
                    df['day'] = df[timestamp_col].dt.day
                    df['hour'] = df[timestamp_col].dt.hour
                    df['dayofweek'] = df[timestamp_col].dt.dayofweek
                    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
                    
                    logger.info(f"Extracted temporal features from {timestamp_col}")
                except Exception as e:
                    logger.error(f"Failed to process timestamps: {e}")
                    raise ValueError(f"Error processing timestamps: {e}")
            
            return df
    
    def clean_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete cleaning pipeline for interaction data"""
        with PipelineLogger(self.logger_name) as logger:
            logger.info(f"Starting data cleaning. Initial shape: {df.shape}")
            
            df = self.remove_duplicates(df, subset=['user_id', 'item_id'])
            
            # Drop unnecessary columns
            df = df.drop(columns=['extra_column'], errors='ignore')
            
            df = self.handle_missing_values(df, strategy={
                'rating': 'drop',
                'user_id': 'drop',
                'item_id': 'drop',
                'timestamp': 'drop',
                'event_type': 'mode'  # Excluded from missing value handling
            })
            
            if 'rating' in df.columns:
                df['rating'] = df['rating'].astype(int)
            
            if 'rating' in df.columns:
                before_filtering = len(df)
                df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
                rows_removed = before_filtering - len(df)
                
                logger.info(f"TEST>>>>>>>>>>>>>>>>>>>>>>>> Filtered invalid ratings. Removed {rows_removed} rows. Remaining rows: {len(df)}")
                
                if df.empty:
                    logger.warning("All rows were removed after filtering invalid ratings. Returning an empty DataFrame.")
            
            df = self.process_timestamps(df, 'timestamp')
            
            
            
            # Ensure the dataset is not empty
            if df.empty:
                logger.warning("All rows were removed during cleaning. Returning an empty DataFrame.")

            
            logger.info(f"Cleaning completed. Final shape: {df.shape}")
            
            saved_path = self.storage.save_processed_data(df, 'interactions_cleaned')
            logger.info(f"Saved cleaned data to: {saved_path}")
            
            return df
    
    def clean_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete cleaning pipeline for product data"""
        with PipelineLogger(self.logger_name) as logger:
            logger.info(f"Starting product cleaning. Initial shape: {df.shape}")
            
            df = self.remove_duplicates(df, subset=['item_id'])
            
            df = self.handle_missing_values(df, strategy={
                'name': 'drop',
                'category': 'mode',
                'price': 'median'
            })
            
            if 'category' in df.columns:
                df = self.encode_categorical(df, ['category'])
            
            if 'price' in df.columns:
                df = self.normalize_numerical(df, ['price'])
            
            logger.info(f"Product cleaning completed. Final shape: {df.shape}")
            
            saved_path = self.storage.save_processed_data(df, 'products_cleaned')
            logger.info(f"Saved cleaned products to: {saved_path}")
            
            return df

if __name__ == "__main__":
    cleaner = DataCleaner()
    
    interactions_df = pd.read_csv('data/raw/user_interactions.csv')
    cleaned_interactions = cleaner.clean_interactions(interactions_df)
    
    print(f"\nCleaned interactions: {cleaned_interactions.shape}")
    print(cleaned_interactions.head())