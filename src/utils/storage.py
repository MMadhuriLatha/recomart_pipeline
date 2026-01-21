import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

class DataLakeStorage:
    """Handles data storage with partitioning structure"""
    
    def __init__(self, base_path='data'):
        self.base_path = Path(base_path)
        self._create_structure()
    
    def _create_structure(self):
        """Create directory structure for data lake"""
        dirs = ['raw', 'processed', 'features', 'models', 'logs']
        for d in dirs:
            (self.base_path / d).mkdir(parents=True, exist_ok=True)
    
    def save_raw_data(self, data, source, data_type, fmt='csv'):
        """
        Save raw data with partitioning by source, date, and type
        
        Args:
            data: DataFrame or dict to save
            source: Data source name
            data_type: Type of data (e.g., 'interactions', 'products')
            fmt: Format ('csv', 'json', 'parquet')
        
        Returns:
            Path to saved file
        """
        date_partition = datetime.now().strftime('%Y/%m/%d')
        partition_path = self.base_path / 'raw' / source / date_partition / data_type
        partition_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{data_type}_{timestamp}.{fmt}'
        filepath = partition_path / filename
        
        if isinstance(data, pd.DataFrame):
            if fmt == 'csv':
                data.to_csv(filepath, index=False)
            elif fmt == 'parquet':
                data.to_parquet(filepath, index=False)
            elif fmt == 'json':
                data.to_json(filepath, orient='records')
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        metadata = {
            'source': source,
            'type': data_type,
            'format': fmt,
            'timestamp': timestamp,
            'path': str(filepath),
            'rows': len(data) if isinstance(data, pd.DataFrame) else None
        }
        
        metadata_file = partition_path / f'{data_type}_{timestamp}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath
    
    def load_latest_raw(self, source, data_type, fmt='csv'):
        """Load the most recent raw data file"""
        raw_path = self.base_path / 'raw' / source
        
        if not raw_path.exists():
            raise FileNotFoundError(f"No data found for source: {source}")
        
        all_files = list(raw_path.rglob(f'{data_type}_*.{fmt}'))
        
        if not all_files:
            raise FileNotFoundError(f"No {fmt} files found for {data_type}")
        
        latest_file = max(all_files, key=lambda x: x.stat().st_mtime)
        
        if fmt == 'csv':
            return pd.read_csv(latest_file)
        elif fmt == 'parquet':
            return pd.read_parquet(latest_file)
        elif fmt == 'json':
            return pd.read_json(latest_file)
    
    def save_processed_data(self, data, name, fmt='parquet'):
        """Save processed data"""
        processed_path = self.base_path / 'processed'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{name}_{timestamp}.{fmt}'
        filepath = processed_path / filename
        
        if fmt == 'csv':
            data.to_csv(filepath, index=False)
        elif fmt == 'parquet':
            data.to_parquet(filepath, index=False)
        
        return filepath
    
    def save_features(self, features, feature_name, version='v1'):
        """Save feature data with versioning"""
        feature_path = self.base_path / 'features' / feature_name / version
        feature_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = feature_path / f'{feature_name}_{timestamp}.parquet'
        
        features.to_parquet(filepath, index=False)
        
        return filepath