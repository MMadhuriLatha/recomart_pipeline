from src.ingestion.csv_ingestion import CSVIngestion
from src.ingestion.api_ingestion import APIIngestion
from src.utils.logger import PipelineLogger
from src.utils.storage import DataLakeStorage

class BatchIngestion:
    """Handles batch ingestion for CSV and API data"""
    
    def __init__(self, storage: DataLakeStorage):
        self.storage = storage
        self.logger_name = 'batch_ingestion'
    
    def run_batch_updates(self):
        """Run batch updates for CSV and API ingestion"""
        with PipelineLogger(self.logger_name) as logger:
            # CSV Batch Update            
            logger.info("Running batch update for CSV data")
            csv_ingestion = CSVIngestion(self.storage)
            csv_ingestion.generate_sample_data()
            interactions_df, interactions_path = csv_ingestion.ingest_file(
                'data/batch/user_interactions.csv',
                source_name='user_behavior_batch',
                data_type='interactions'
            )
            logger.info(f"Batch updated interactions: {len(interactions_df)} records")
            logger.info(f"Saved to: {interactions_path}")
            
            # API Batch Update
            logger.info("Running batch update for API data")
            api_ingestion = APIIngestion(self.storage)
            url = "https://fakestoreapi.com/products"
        
            # Ingest data from the API
            products_df, products_path = api_ingestion.ingest_api_data(
                url=url,
                source_name='product_api',
                data_type='catalog'
            )
            
            # Rename columns to match the expected schema
            products_df.rename(columns={
                'id': 'item_id',
                'title': 'name',
                'price': 'price',
                'category': 'category',
                'description': 'description',
                'image': 'image_url'
            }, inplace=True)
            
            # Extract rating details
            if 'rating' in products_df.columns:
                products_df['rating_score'] = products_df['rating'].apply(lambda x: x.get('rate', 0) if isinstance(x, dict) else 0)
                products_df['rating_count'] = products_df['rating'].apply(lambda x: x.get('count', 0) if isinstance(x, dict) else 0)
                products_df.drop('rating', axis=1, inplace=True)
            logger.info(f"Batch updated products: {len(products_df)} records")
            logger.info(f"Saved to: {products_path}")
            
            # Generate the features column
            products_df['features'] = products_df.apply(
                lambda row: f"{row['name']} {row['category']} {row['description']}".lower().replace('\n', ' ').strip(),
                axis=1
            )
            
            return interactions_df, products_df