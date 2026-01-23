from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import pendulum
from datetime import datetime, timedelta
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  

sys.path.append(str(str(PROJECT_ROOT)))

# sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.csv_ingestion import CSVIngestion
from src.ingestion.api_ingestion import APIIngestion
from src.validation.data_validator import DataValidator
from src.preparation.data_cleaner import DataCleaner
from src.features.feature_engineer import FeatureEngineer
from src.features.feature_store import FeatureStore
from src.models.recommender import RecommenderTrainer
from src.utils.storage import DataLakeStorage
import pandas as pd

default_args = {
    'owner': 'recomart',
    'depends_on_past': False,
    'start_date': pendulum.datetime(2024, 1, 1, tz="UTC"),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'recomart_recommendation_pipeline',
    default_args=default_args,
    description='End-to-end recommendation system pipeline',
    schedule='@daily',
    catchup=False,
    tags=['recommendation', 'ml', 'data-pipeline']
)

def ingest_csv_data(**context):
    """Task: Ingest CSV interaction data"""
    print("Starting CSV data ingestion...")
    
    ingestion = CSVIngestion()
    
    sample_df = ingestion.generate_sample_data()
    
    df, saved_path = ingestion.ingest_file(
        'data/raw/user_interactions.csv',
        source_name='user_behavior',
        data_type='interactions'
    )
    
    context['ti'].xcom_push(key='interactions_count', value=len(df))
    context['ti'].xcom_push(key='interactions_path', value=str(saved_path))
    
    print(f"Ingested {len(df)} interactions")
    return str(saved_path)

def ingest_api_data(**context):
    """Task: Ingest product catalog from API"""
    print("Starting API data ingestion...")
    
    ingestion = APIIngestion()
    df, saved_path = ingestion.ingest_product_catalog()
    
    context['ti'].xcom_push(key='products_count', value=len(df))
    context['ti'].xcom_push(key='products_path', value=str(saved_path))
    
    print(f"Ingested {len(df)} products")
    return str(saved_path)

def validate_data(**context):
    """Task: Validate data quality"""
    print("Starting data validation...")
    
    storage = DataLakeStorage()
    df = storage.load_latest_raw('user_behavior', 'interactions', fmt='csv')
    
    validator = DataValidator()
    results = validator.validate_interactions(df)
    validator.generate_report('reports/data_quality_report.json')
    
    context['ti'].xcom_push(key='quality_score', value=results['quality_score'])
    
    if results['quality_score'] < 70:
        raise ValueError(f"Data quality score too low: {results['quality_score']}")
    
    print(f"Data quality score: {results['quality_score']}/100")
    return results['quality_score']

def clean_data(**context):
    """Task: Clean and prepare data"""
    print("Starting data cleaning...")
    
    storage = DataLakeStorage()
    interactions_df = storage.load_latest_raw('user_behavior', 'interactions', fmt='csv')
    
    cleaner = DataCleaner(storage)
    cleaned_df = cleaner.clean_interactions(interactions_df)
    
    context['ti'].xcom_push(key='cleaned_count', value=len(cleaned_df))
    
    print(f"Cleaned data: {len(cleaned_df)} records")
    return len(cleaned_df)

def engineer_features(**context):
    """Task: Create features"""
    print("Starting feature engineering...")
    
    storage = DataLakeStorage()
    interactions_df = pd.read_parquet(
        list(Path('data/processed').glob('interactions_cleaned_*.parquet'))[0]
    )
    
    engineer = FeatureEngineer(storage)
    features = engineer.engineer_all_features(interactions_df)
    
    context['ti'].xcom_push(key='user_features_count', value=len(features['user_features']))
    context['ti'].xcom_push(key='item_features_count', value=len(features['item_features']))
    
    print(f"Created features for {len(features['user_features'])} users")
    print(f"Created features for {len(features['item_features'])} items")
    
    return True

def save_to_feature_store(**context):
    """Task: Save features to feature store"""
    print("Saving to feature store...")
    
    feature_store = FeatureStore()
    
    user_features = pd.read_parquet(
        list(Path('data/features/user_features/v1').glob('*.parquet'))[0]
    )
    item_features = pd.read_parquet(
        list(Path('data/features/item_features/v1').glob('*.parquet'))[0]
    )
    
    feature_store.register_feature(
        feature_name='user_features',
        description='User-level aggregated features',
        feature_type='user',
        source='user_interactions',
        transformation='Aggregation by user_id',
        version='v1'
    )
    
    feature_store.register_feature(
        feature_name='item_features',
        description='Item-level aggregated features',
        feature_type='item',
        source='user_interactions',
        transformation='Aggregation by item_id',
        version='v1'
    )
    
    print("Features registered in feature store")
    return True

def train_model(**context):
    """Task: Train recommendation model"""
    print("Starting model training...")
    
    df = pd.read_parquet(
        list(Path('data/processed').glob('interactions_cleaned_*.parquet'))[0]
    )
    
    trainer = RecommenderTrainer()
    results = trainer.train_and_evaluate(
        df,
        n_factors=20,
        test_size=0.2,
        random_state=42
    )
    
    context['ti'].xcom_push(key='rmse', value=results['metrics']['rmse'])
    context['ti'].xcom_push(key='mae', value=results['metrics']['mae'])
    
    print(f"Model trained - RMSE: {results['metrics']['rmse']:.4f}")
    print(f"Model trained - MAE: {results['metrics']['mae']:.4f}")
    
    return results['metrics']

def pipeline_summary(**context):
    """Task: Generate pipeline summary"""
    ti = context['ti']
    
    summary = {
        'execution_date': str(context.get('logical_date')),
        'interactions_ingested': ti.xcom_pull(key='interactions_count', task_ids='ingest_csv'),
        'products_ingested': ti.xcom_pull(key='products_count', task_ids='ingest_api'),
        'quality_score': ti.xcom_pull(key='quality_score', task_ids='validate'),
        'cleaned_records': ti.xcom_pull(key='cleaned_count', task_ids='clean'),
        'rmse': ti.xcom_pull(key='rmse', task_ids='train'),
        'mae': ti.xcom_pull(key='mae', task_ids='train')
    }
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("="*60)
    
    return summary

ingest_csv_task = PythonOperator(
    task_id='ingest_csv',
    python_callable=ingest_csv_data,
    dag=dag
)

ingest_api_task = PythonOperator(
    task_id='ingest_api',
    python_callable=ingest_api_data,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate',
    python_callable=validate_data,
    dag=dag
)

clean_task = PythonOperator(
    task_id='clean',
    python_callable=clean_data,
    dag=dag
)

feature_engineering_task = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features,
    dag=dag
)

feature_store_task = PythonOperator(
    task_id='save_to_feature_store',
    python_callable=save_to_feature_store,
    dag=dag
)

train_task = PythonOperator(
    task_id='train',
    python_callable=train_model,
    dag=dag
)

summary_task = PythonOperator(
    task_id='pipeline_summary',
    python_callable=pipeline_summary,
    dag=dag
)

[ingest_csv_task, ingest_api_task] >> validate_task >> clean_task
clean_task >> feature_engineering_task >> feature_store_task >> train_task
train_task >> summary_task