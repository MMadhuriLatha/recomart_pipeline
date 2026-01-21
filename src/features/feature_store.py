import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys
import shutil
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import PipelineLogger

class FeatureStore:
    """
    Simple feature store for managing, versioning, and retrieving features
    """
    
    def __init__(self, base_path: str = 'data/features'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_path / 'feature_metadata.json'
        self.metadata = self._load_metadata()
        self.logger_name = 'feature_store'
    
    def _load_metadata(self) -> Dict:
        """Load feature metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'features': {}}
    
    def _save_metadata(self):
        """Save feature metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_feature(self, feature_name: str, description: str,
                        feature_type: str, source: str,
                        transformation: str, version: str = 'v1'):
        """
        Register a new feature in the metadata registry
        
        Args:
            feature_name: Name of the feature
            description: Description of what the feature represents
            feature_type: Type of feature (user, item, interaction)
            source: Source data/table
            transformation: Description of transformation applied
            version: Feature version
        """
        with PipelineLogger(self.logger_name) as logger:
            feature_key = f"{feature_name}_{version}"
            
            self.metadata['features'][feature_key] = {
                'name': feature_name,
                'description': description,
                'type': feature_type,
                'source': source,
                'transformation': transformation,
                'version': version,
                'registered_at': datetime.now().isoformat(),
                'status': 'active'
            }
            
            self._save_metadata()
            logger.info(f"Registered feature: {feature_key}")
    
    def save_feature(self, df: pd.DataFrame, feature_name: str, 
                    version: str = 'v1', description: str = '',
                    feature_type: str = '', source: str = '',
                    transformation: str = ''):
        """
        Save feature data and register metadata
        
        Args:
            df: Feature DataFrame
            feature_name: Name of the feature set
            version: Version identifier
            description: Feature description
            feature_type: Type of feature
            source: Source of feature
            transformation: Transformation description
        """
        with PipelineLogger(self.logger_name) as logger:
            feature_path = self.base_path / feature_name / version
            feature_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{feature_name}_{timestamp}.parquet"
            filepath = feature_path / filename
            
            df.to_parquet(filepath, index=False)
            
            latest_link = feature_path / f"{feature_name}_latest.parquet"
            if latest_link.exists():
                latest_link.unlink()
            latest_link = feature_path / f"{feature_name}_latest.parquet"
            shutil.copy2(filepath, latest_link)
            
            self.register_feature(
                feature_name=feature_name,
                description=description,
                feature_type=feature_type,
                source=source,
                transformation=transformation,
                version=version
            )
            
            logger.info(f"Saved feature {feature_name} version {version}")
            logger.info(f"Shape: {df.shape}, Path: {filepath}")
            
            return filepath
    
    def get_feature(self, feature_name: str, version: str = 'v1',
                   use_latest: bool = True) -> pd.DataFrame:
        """
        Retrieve feature data
        
        Args:
            feature_name: Name of feature set
            version: Version to retrieve
            use_latest: Whether to use the latest version
        
        Returns:
            Feature DataFrame
        """
        with PipelineLogger(self.logger_name) as logger:
            feature_path = self.base_path / feature_name / version
            
            if not feature_path.exists():
                raise FileNotFoundError(f"Feature {feature_name} version {version} not found")
            
            if use_latest:
                latest_file = feature_path / f"{feature_name}_latest.parquet"
                if latest_file.exists():
                    df = pd.read_parquet(latest_file)
                    logger.info(f"Loaded latest {feature_name} v{version}: {df.shape}")
                    return df
            
            files = list(feature_path.glob(f"{feature_name}_*.parquet"))
            if not files:
                raise FileNotFoundError(f"No feature files found for {feature_name}")
            
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            df = pd.read_parquet(latest_file)
            
            logger.info(f"Loaded {feature_name} v{version}: {df.shape}")
            return df
    
    def list_features(self, feature_type: Optional[str] = None) -> pd.DataFrame:
        """
        List all registered features
        
        Args:
            feature_type: Filter by feature type
        
        Returns:
            DataFrame of feature metadata
        """
        features = []
        
        for key, meta in self.metadata['features'].items():
            if feature_type is None or meta['type'] == feature_type:
                features.append(meta)
        
        return pd.DataFrame(features)
    
    def get_feature_metadata(self, feature_name: str, version: str = 'v1') -> Dict:
        """Get metadata for a specific feature"""
        feature_key = f"{feature_name}_{version}"
        
        if feature_key not in self.metadata['features']:
            raise KeyError(f"Feature {feature_key} not found in metadata")
        
        return self.metadata['features'][feature_key]
    
    def create_feature_view(self, feature_names: List[str], 
                           join_key: str, version: str = 'v1') -> pd.DataFrame:
        """
        Create a feature view by joining multiple feature sets
        
        Args:
            feature_names: List of feature set names
            join_key: Column to join on (e.g., 'user_id', 'item_id')
            version: Version to use
        
        Returns:
            Joined DataFrame
        """
        with PipelineLogger(self.logger_name) as logger:
            logger.info(f"Creating feature view with {len(feature_names)} feature sets")
            
            result = None
            
            for fname in feature_names:
                df = self.get_feature(fname, version)
                
                if result is None:
                    result = df
                else:
                    result = result.merge(df, on=join_key, how='left')
                
                logger.info(f"Added {fname}: {result.shape}")
            
            logger.info(f"Feature view created: {result.shape}")
            return result

if __name__ == "__main__":
    store = FeatureStore()
    
    user_features = pd.DataFrame({
        'user_id': [1, 2, 3],
        'avg_rating': [4.5, 3.2, 4.8],
        'interaction_count': [10, 5, 15]
    })
    
    store.save_feature(
        user_features,
        feature_name='user_features',
        version='v1',
        description='User-level aggregated features',
        feature_type='user',
        source='user_interactions',
        transformation='Aggregation by user_id'
    )
    
    print("\nRegistered Features:")
    print(store.list_features())
    
    loaded_features = store.get_feature('user_features', 'v1')
    print(f"\nLoaded features: {loaded_features.shape}")
    print(loaded_features.head())