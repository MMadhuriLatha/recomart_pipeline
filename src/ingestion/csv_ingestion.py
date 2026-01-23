import pandas as pd
import numpy as np
from pathlib import Path
import time
from typing import Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import PipelineLogger
from src.utils.storage import DataLakeStorage

class CSVIngestion:
    """Handles CSV data ingestion with error handling and retries"""
    
    def __init__(self, storage: Optional[DataLakeStorage] = None):
        self.storage = storage or DataLakeStorage()
        self.logger_name = 'csv_ingestion'
    
    def ingest_file(self, filepath, source_name, data_type, max_retries=3):
        """
        Ingest CSV file with retry mechanism
        
        Args:
            filepath: Path to CSV file
            source_name: Name of data source
            data_type: Type of data being ingested
            max_retries: Maximum number of retry attempts
        
        Returns:
            DataFrame and path to saved file
        """
        with PipelineLogger(self.logger_name) as logger:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting to ingest {filepath} (attempt {attempt + 1})")
                    
                    df = pd.read_csv(filepath)
                    
                    logger.info(f"Successfully loaded {len(df)} rows from {filepath}")
                    logger.info(f"Columns: {list(df.columns)}")
                    logger.info(f"Shape: {df.shape}")
                    
                    saved_path = self.storage.save_raw_data(
                        df, 
                        source=source_name,
                        data_type=data_type,
                        fmt='csv'
                    )
                    
                    logger.info(f"Saved raw data to: {saved_path}")
                    
                    return df, saved_path
                
                except FileNotFoundError as e:
                    logger.error(f"File not found: {filepath}")
                    raise
                
                except pd.errors.EmptyDataError as e:
                    logger.error(f"Empty CSV file: {filepath}")
                    raise
                
                except Exception as e:
                    logger.error(f"Error ingesting file (attempt {attempt + 1}): {str(e)}")
                    
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error("Max retries reached. Ingestion failed.")
                        raise
    
    def generate_sample_data(self):
        """Generate sample user interaction data for testing"""
        with PipelineLogger(self.logger_name) as logger:
            logger.info("Generating sample user interaction data")
            
            np.random.seed(42)
            
            n_users = 1000
            n_items = 500
            n_interactions = 10000
            
            data = {
                'user_id': np.random.randint(1, n_users + 1, n_interactions),
                'item_id': np.random.randint(1, n_items + 1, n_interactions),
                'rating': np.random.randint(1, 6, n_interactions),
                'timestamp': pd.date_range('2024-01-01', periods=n_interactions, freq='1H')
            }
            
            df = pd.DataFrame(data)
            
            df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
            
            sample_path = Path('data/raw/user_interactions.csv')
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(sample_path, index=False)
            
            logger.info(f"Generated {len(df)} sample interactions")
            logger.info(f"Saved to: {sample_path}")
            
            return df

if __name__ == "__main__":
    ingestion = CSVIngestion()
    
    sample_df = ingestion.generate_sample_data()
    
    df, saved_path = ingestion.ingest_file(
        'data/raw/user_interactions.csv',
        source_name='user_behavior',
        data_type='interactions'
    )
    
    print(f"Ingested {len(df)} interactions")
    print(f"Saved to: {saved_path}")
