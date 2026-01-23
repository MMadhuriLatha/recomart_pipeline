"""
RecoMart Recommendation Pipeline - Main Execution Script
Run the complete end-to-end pipeline
"""

import sys
import time
import threading
import queue
from pathlib import Path
import pandas as pd
import warnings
import subprocess
warnings.filterwarnings('ignore')

from src.ingestion.csv_ingestion import CSVIngestion
from src.ingestion.api_ingestion import APIIngestion
from src.ingestion.streaming_ingestion import StreamingIngestion
from src.ingestion.batch_ingestion import BatchIngestion
from src.validation.data_validator import DataValidator
from src.preparation.data_cleaner import DataCleaner
from src.features.feature_engineer import FeatureEngineer
from src.features.feature_store import FeatureStore
from src.models.recommender import RecommenderTrainer
from src.utils.logger import PipelineLogger
from src.utils.storage import DataLakeStorage

# ==============================
# GLOBAL SHARED STATE
# ==============================
interaction_queue = queue.Queue()
latest_products = None
stop_event = threading.Event()
product_ready_event = threading.Event()

def print_header(message):
    print("\n" + "=" * 70)
    print(f"  {message}")
    print("=" * 70 + "\n")


def process_streaming_data(streaming_data):
    """
    Process streaming data and add a 'rating' column based on event_type.
    """
    event_to_rating = {
        'click': 1,
        'add_to_cart': 3,
        'purchase': 5
    }
    
    streaming_data['rating'] = streaming_data['event_type'].map(event_to_rating).fillna(0)
    return streaming_data

# ==============================
# STREAMING WORKER
# ==============================
def streaming_worker(storage):
    streaming = StreamingIngestion(storage)

    while not stop_event.is_set():
        try:
            df = streaming.ingest_streaming_data(
                queue_name='user_interactions',
                rabbitmq_host='localhost',
                max_messages=50
            )
            df = process_streaming_data(df)
            if df is not None and not df.empty:
                interaction_queue.put(df)
                print(f"[Streaming] Pushed {len(df)} records to pipeline")

        except Exception as e:
            print(f"[Streaming Error] {e}")

        time.sleep(2)


# ==============================
# BATCH WORKER
# ==============================
def batch_worker(storage, interval_minutes=10):
    global latest_products
    batch = BatchIngestion(storage)
    batch_file = Path("data/batch/user_interactions.csv")
    
    while not stop_event.is_set():
        try:
            if not batch_file.exists():
                print("[Batch] Waiting for batch file to appear...")
                time.sleep(60)
                continue

            batch_df, product_df = batch.run_batch_updates()
            
            # Check if product_df is empty
            if product_df.empty:
                print("Product DataFrame is empty. Skipping further processing.")
                return  # Skip this iteration

            if batch_df is not None and not batch_df.empty:
                interaction_queue.put(batch_df)
                latest_products = product_df
                product_ready_event.set()
                print(f"[Batch] Ingested {len(batch_df)} records")

        except Exception as e:
            print(f"[Batch Error] {e}")

        time.sleep(interval_minutes * 60)


# ==============================
# MAIN PIPELINE ORCHESTRATOR
# ==============================
def orchestrator():
    global latest_products

    storage = DataLakeStorage()
    validator = DataValidator()
    cleaner = DataCleaner(storage)
    engineer = FeatureEngineer(storage)
    feature_store = FeatureStore()
    trainer = RecommenderTrainer()

    print_header("RECOMART RECOMMENDATION PIPELINE - STARTED")
    
    # Wait for product_df to be ready
    print("[Orchestrator] Waiting for product catalog to be ingested...")
    product_ready_event.wait()  # Wait until product_ready_event is set
    print("[Orchestrator] Product catalog ingestion completed. Starting pipeline.")

    combined_interactions = []  # List to store all batches

    while not interaction_queue.empty():
        batch = interaction_queue.get()  # Get one batch from the queue
        combined_interactions.append(batch)  # Add the batch to the list

    while not stop_event.is_set():
        # Retrieve a batch from the queue
        # interactions = interaction_queue.get()
        # combined_interactions.append(interactions)  # Add the batch to the list

        # Combine all batches into a single DataFrame
        all_interactions = pd.concat(combined_interactions, ignore_index=True)
        print(f"Total rows in combined dataset: {len(all_interactions)}")


        # # ==============================
        # # TASK 4: DATA VALIDATION
        # # ==============================
        # print_header("TASK 4: DATA VALIDATION")
        # validation_results = validator.validate_interactions(interactions)
        # validator.generate_report('reports/data_quality_report.json')

        # print(f"✓ Data quality score: {validation_results['quality_score']}/100")
        # print(f"  Missing values: {validation_results['missing_values']['total_missing']}")
        # print(f"  Duplicate records: {validation_results['duplicates']['total_duplicates']}")
        # print(f"  Unique users: {validation_results['statistics']['unique_users']}")
        # print(f"  Unique items: {validation_results['statistics']['unique_items']}")

        # if validation_results['quality_score'] < 70:
        #     print("⚠ WARNING: Quality score below threshold!")
        #     continue
        
        # ==============================
        # TASK 4: DATA VALIDATION (BEFORE CLEANING)
        # ==============================
        print_header("TASK 4: DATA VALIDATION (BEFORE CLEANING)")
        validation_results_before = validator.validate_interactions(all_interactions)
        print("validation_results_before:", validation_results_before)
        validator.generate_report('reports/data_quality_report_before_cleaning.json')
        validator.generate_pdf_report("reports/data_quality_report_before_cleaning.pdf"
        )

        print(f"✓ Data quality score (before cleaning): {validation_results_before['quality_score']}/100")
        print(f"  Missing values: {validation_results_before['missing_values']['total_missing']}")
        print(f"  Duplicate records: {validation_results_before['duplicates']['total_duplicates']}")
        print(f"  Unique users: {validation_results_before['statistics']['unique_users']}")
        print(f"  Unique items: {validation_results_before['statistics']['unique_items']}")

        if validation_results_before['quality_score'] < 70:
            print("⚠ WARNING: Quality score below threshold! Proceeding with cleaning...")

        print("Columns before cleaning:", all_interactions.columns)

        # ==============================
        # TASK 5: DATA PREPARATION & CLEANING
        # ==============================
        print_header("TASK 5: DATA PREPARATION & CLEANING")

        cleaned_interactions = cleaner.clean_interactions(all_interactions)

        if latest_products is not None:
            cleaned_products = cleaner.clean_products(latest_products)
        else:
            cleaned_products = pd.DataFrame()

        print(f"✓ Cleaned interactions: {len(cleaned_interactions)} records")
        print("Columns after cleaning:", cleaned_interactions.columns)
        print(f"✓ Cleaned products: {len(cleaned_products)} records")
        print("Columns after cleaning:", cleaned_products.columns)


        # ==============================
        # TASK 4: DATA VALIDATION (AFTER CLEANING)
        # ==============================
        print_header("TASK 4: DATA VALIDATION (AFTER CLEANING)")
        validation_results_after = validator.validate_interactions(cleaned_interactions)
        print("validation_results_after:", validation_results_after)
        validator.generate_report('reports/data_quality_report_after_cleaning.json')
        validator.generate_pdf_report("reports/data_quality_report_after_cleaning.pdf"
        )
        print(f"✓ Data quality score (after cleaning): {validation_results_after['quality_score']}/100")
        print(f"  Missing values: {validation_results_after['missing_values']['total_missing']}")
        print(f"  Duplicate records: {validation_results_after['duplicates']['total_duplicates']}")
        print(f"  Unique users: {validation_results_after['statistics']['unique_users']}")
        print(f"  Unique items: {validation_results_after['statistics']['unique_items']}")

        # ==============================
        # TASK 6: FEATURE ENGINEERING
        # ==============================
        print_header("TASK 6: FEATURE ENGINEERING")
        features = engineer.engineer_all_features(cleaned_interactions, cleaned_products)

        print(f"✓ User features: {features['user_features'].shape}")
        print(f"  - interaction_count, avg_rating, rating_std, etc.")
        print(f"✓ Item features: {features['item_features'].shape}")
        print(f"  - popularity_score, avg_item_rating, etc.")
        print(f"✓ Interaction features: {features['interaction_features'].shape}")

        # ==============================
        # TASK 7: FEATURE STORE
        # ==============================
        print_header("TASK 7: FEATURE STORE")
        feature_store.save_feature(
            features['user_features'],
            feature_name='user_features',
            version='v1',
            description='User-level aggregated features',
            feature_type='user',
            source='user_interactions',
            transformation='Aggregation and statistical features by user_id'
        )
        print("Saved user features to feature store")

        feature_store.save_feature(
            features['item_features'],
            feature_name='item_features',
            version='v1',
            description='Item-level aggregated features',
            feature_type='item',
            source='user_interactions',
            transformation='Aggregation and popularity calculations by item_id'
        )
        print("Saved item features to feature store")
        
        print("\nRegistered Features:")
        print(feature_store.list_features())

       # ==============================
        # TASK 8: DATA VERSIONING
        # ==============================
        print_header("TASK 8: DATA VERSIONING")
        print("DVC Configuration:")

        try:
            # Initialize DVC (if not already initialized)
            if not Path(".dvc").exists():
                print("✓ Initialize DVC: dvc init")
                subprocess.run(["dvc", "init"], check=True)
                subprocess.run(["dvc", "config", "core.analytics", "false"], check=False)
            
            # Configure Local Remote Storage
            # Create the local storage directory
            Path("dvc_storage").mkdir(exist_ok=True)
            
            # Add the remote (check=False allows this to pass if remote is already added)
            print("Configure remote: dvc remote add -d project_storage ./dvc_storage")
            subprocess.run(
                ["dvc", "remote", "add", "-d", "project_storage", "./dvc_storage"], 
                check=False, 
                capture_output=True # Suppress "remote already exists" errors in logs
            )

            # Add data folders to DVC tracking
            print("Add data: dvc add data/raw data/processed data/features")
            subprocess.run(
                ["dvc", "add", "data/raw", "data/processed", "data/features"], 
                check=True
            )

            # Commit changes to Git (Includes the new remote config)
            print("Commit: git add . && git commit -m 'Add data versions'")
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(
                ["git", "commit", "-m", "Add data versions and configure remote"], 
                check=False  # Don't crash if nothing new to commit
            )

            # Push data to the local remote (NEW STEP)
            print("Track changes: dvc push")
            subprocess.run(["dvc", "push"], check=True)
            print("  -> Data successfully pushed to ./dvc_storage")

        except subprocess.CalledProcessError as e:
            print(f"DVC Error: Command failed with exit code {e.returncode}")
        except Exception as e:
            print(f"Unexpected Error during versioning: {e}")


        # ==============================
        # TASK 9: MODEL TRAINING & EVALUATION
        # ==============================
        print_header("TASK 9: MODEL TRAINING & EVALUATION")
        # results = trainer.train_and_evaluate(
        #     cleaned_interactions,
        #     n_factors=50,
        #     test_size=0.2,
        #     random_state=42
        # )

        # print(f"✓ Model trained successfully")
        # print(f"  RMSE: {results['metrics']['rmse']:.4f}")
        # print(f"  MAE: {results['metrics']['mae']:.4f}")
        # print(f"  Train samples: {len(results['train_df'])}")
        # print(f"  Test samples: {len(results['test_df'])}")
        
        # Train collaborative filtering model
        results_cf = trainer.train_and_evaluate(user_interactions_df=cleaned_interactions, n_factors=20, test_size=0.2, model_type='collaborative')
        print(f"\nCollaborative Filtering Metrics:")
        print(f"RMSE: {results_cf['metrics']['rmse']:.4f}")
        print(f"MAE: {results_cf['metrics']['mae']:.4f}")

        # Train content-based filtering model
        results_cb = trainer.train_and_evaluate(product_catalog_df=cleaned_products, model_type='content_based')
        print(f"\nSample Content-Based Recommendations:")
        print(results_cb['recommendations'])
        
        print_header("GENERATING SAMPLE RECOMMENDATIONS")
        
        # Validate data for collaborative filtering
        required_columns_cf = {'user_id', 'item_id', 'rating'}
        if not required_columns_cf.issubset(cleaned_interactions.columns):
            raise ValueError(f"Missing required columns for collaborative filtering: {required_columns_cf - set(cleaned_interactions.columns)}")

        # Validate data for content-based filtering
        required_columns_cb = {'item_id', 'features'}
        if not required_columns_cb.issubset(cleaned_products.columns):
            raise ValueError(f"Missing required columns for content-based filtering: {required_columns_cb - set(cleaned_products.columns)}")

        # Collaborative Filtering Recommendations
        sample_user = cleaned_interactions['user_id'].iloc[0]
        user_history = set(
            cleaned_interactions[cleaned_interactions['user_id'] == sample_user]['item_id']
        )
        
        collaborative_recommendations = results_cf['model'].recommend_items(
            user_id=sample_user,
            n_recommendations=10,
            exclude_known=True,
            known_items=user_history
        )

        print(f"\nTop 10 Collaborative Filtering Recommendations for User {sample_user}:")
        print(collaborative_recommendations)

        # Content-Based Recommendations
        content_based_recommendations = results_cb['model'].recommend_items(
            user_history=list(user_history),
            n_recommendations=10
        )

        print(f"\nTop 10 Content-Based Recommendations for User {sample_user}:")
        print(content_based_recommendations)
        
        
        print_header("PIPELINE EXECUTION SUMMARY")

        summary = {
            'Data Quality Score (After Cleaning)': f"{validation_results_after['quality_score']}/100",
            'Total Interactions Processed': len(cleaned_interactions),
            'Total Products Processed': len(cleaned_products),
            'Unique Users': cleaned_interactions['user_id'].nunique(),
            'Unique Items': cleaned_interactions['item_id'].nunique(),
            'Collaborative Model RMSE': f"{results_cf['metrics']['rmse']:.4f}",
            'Collaborative Model MAE': f"{results_cf['metrics']['mae']:.4f}",
            'Content-Based Model Precision': f"{results_cb['metrics']['precision']:.4f}",
            'Content-Based Model Recall': f"{results_cb['metrics']['recall']:.4f}",
            'Content-Based Model Coverage': f"{results_cb['metrics']['coverage']:.4f}",
            'Features in Store': len(feature_store.list_features()),
            'Content-Based Recommendations Generated': len(content_based_recommendations),
            'Collaborative Recommendations Generated': len(collaborative_recommendations)
        }

        for key, value in summary.items():
            print(f"{key:.<50} {value}") 
                   
        print_header("PIPELINE COMPLETED SUCCESSFULLY ✓")
    
        print("\nNext Steps:")
        print("1. Review data quality report: reports/data_quality_report.json")
        print("2. Check MLflow experiments: mlflow ui")
        print("3. Set up Airflow DAG: orchestration/airflow_dag.py")
        print("4. Version data with DVC")
        print("5. Deploy model for inference")
    
        return {
            'interactions': cleaned_interactions,
            'products': cleaned_products,
            'features': features,
            'model': results_cf['model'],
            'metrics': results_cf['metrics'],
            'feature_store': feature_store
        }

        # print_header("PIPELINE ITERATION COMPLETED")
        # print("Waiting for the next iteration...\n")


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    try:
        storage = DataLakeStorage()
        
        print_header("RECOMART RECOMMENDATION PIPELINE - STARTED")
        
        # ==============================
        # TASK 2 & 3: DATA INGESTION & RAW DATA STORAGE
        # ==============================
        
        print_header("TASK 2 & 3: DATA INGESTION & RAW DATA STORAGE")
    
        # Start the worker threads
        threading.Thread(target=streaming_worker, args=(storage,), daemon=True).start()
        threading.Thread(target=batch_worker, args=(storage,), daemon=True).start()

        # Start the orchestrator
        orchestrator()

    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")
        stop_event.set()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {str(e)}")
        stop_event.set()
        sys.exit(1)
