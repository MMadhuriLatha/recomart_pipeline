import requests
import pandas as pd
import time
from typing import Optional, Dict, List
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import PipelineLogger
from src.utils.storage import DataLakeStorage

class APIIngestion:
    """Handles REST API data ingestion with error handling and retries"""
    
    def __init__(self, storage: Optional[DataLakeStorage] = None):
        self.storage = storage or DataLakeStorage()
        self.logger_name = 'api_ingestion'
    
    def fetch_data(self, url: str, headers: Optional[Dict] = None, 
                   params: Optional[Dict] = None, max_retries: int = 3,
                   timeout: int = 30) -> List[Dict]:
        """
        Fetch data from REST API with retry mechanism
        
        Args:
            url: API endpoint URL
            headers: Request headers
            params: Query parameters
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        
        Returns:
            List of records
        """
        with PipelineLogger(self.logger_name) as logger:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Fetching data from {url} (attempt {attempt + 1})")

                    try:
                        response = requests.get(
                            url,
                            headers=headers,
                            params=params,
                            timeout=timeout
                        )
                    except Exception as e:
                        print(f"Exception occurred during requests.get(): {e}")
                        raise
                    
                    logger.info(f"Response status code: {response.status_code}")
                    # Check if the response is valid
                    if response.status_code != 200:
                        logger.error(f"Unexpected status code: {response.status_code}")
                        raise requests.exceptions.HTTPError(f"HTTP {response.status_code}")
                    
                    # Parse the JSON response
                    try:
                        data = response.json()
                    except ValueError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        raise
                    
                    # Check if the response contains data
                    if not data:
                        logger.warning("API returned an empty response.")
                        raise ValueError("Empty response from API.")
                    
                    logger.info(f"Successfully fetched {len(data) if isinstance(data, list) else 1} records")
                    
                    return data
                
                except requests.exceptions.Timeout:
                    logger.error(f"Request timeout after {timeout}s")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise
                
                except requests.exceptions.HTTPError as e:
                    logger.error(f"HTTP error: {e.response.status_code}")
                    if e.response.status_code >= 500 and attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise
                
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Unhandled exception: {str(e)}")
                    raise
    
    def ingest_api_data(self, url: str, source_name: str, data_type: str,
                        headers: Optional[Dict] = None,
                        params: Optional[Dict] = None) -> tuple:
        """
        Ingest data from API and save to data lake
        
        Args:
            url: API endpoint
            source_name: Name of data source
            data_type: Type of data
            headers: Request headers
            params: Query parameters
        
        Returns:
            DataFrame and saved file path
        """
        with PipelineLogger(self.logger_name) as logger:
            data = self.fetch_data(url, headers, params)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
                
            if df.empty:
                logger.warning("API returned an empty DataFrame.")
                return df, None
            
            logger.info(f"Converted to DataFrame: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            saved_path = self.storage.save_raw_data(
                df,
                source=source_name,
                data_type=data_type,
                fmt='json'
            )
            
            logger.info(f"Saved API data to: {saved_path}")
            
            return df, saved_path
    
    def ingest_product_catalog(self):
        """Ingest product catalog from Fake Store API"""
        url = "https://fakestoreapi.com/products"
        
        # Ingest data from the API
        df, saved_path = self.ingest_api_data(
            url=url,
            source_name='product_api',
            data_type='catalog'
        )
        
        # Rename columns for consistency
        df = df.rename(columns={
            'id': 'item_id',
            'title': 'name',
            'price': 'price',
            'category': 'category',
            'description': 'description',
            'image': 'image_url'
        })
        
        # Extract rating details if available
        if 'rating' in df.columns:
            df['rating_score'] = df['rating'].apply(lambda x: x.get('rate', 0) if isinstance(x, dict) else 0)
            df['rating_count'] = df['rating'].apply(lambda x: x.get('count', 0) if isinstance(x, dict) else 0)
            df = df.drop('rating', axis=1)
            
        # Generate the features column
        df['features'] = df.apply(
            lambda row: f"{row['name']} {row['category']} {row['description']}".lower().replace('\n', ' ').strip(),
            axis=1
        )
        
        return df, saved_path

if __name__ == "__main__":
    ingestion = APIIngestion()
    
    df, saved_path = ingestion.ingest_product_catalog()
    
    print(f"\nIngested {len(df)} products")
    print(f"Saved to: {saved_path}")
    print(f"\nSample data:")
    print(df.head())