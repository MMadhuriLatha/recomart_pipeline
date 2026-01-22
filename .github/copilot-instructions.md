# Copilot Instructions for RecoMart Recommendation Pipeline

## Project Overview
RecoMart is an end-to-end data engineering pipeline for a recommendation system. It processes user behavior data and product catalogs through ingestion, validation, preparation, feature engineering, and model training stages, using MLflow for tracking and Airflow for orchestration.

## Architecture & Data Flow

### Core Pipeline Stages (Sequential Execution)
The pipeline follows a strict sequential flow in `main_pipeline.py`:
1. **Data Ingestion** → Batch/Streaming ingestion (CSV/API/RabbitMQ)
2. **Data Validation** → Quality checks using `DataValidator`
3. **Data Preparation** → Cleaning with `DataCleaner`
4. **Feature Engineering** → Feature generation with `FeatureEngineer`
5. **Feature Store** → Versioning and metadata in `FeatureStore`
6. **Model Training** → SVD matrix factorization with MLflow tracking

### Multi-Source Data Ingestion Pattern
The system handles **three parallel data flows** with worker threads:
- **Streaming**: RabbitMQ-based real-time ingestion (`StreamingIngestion` + `streaming_worker`)
- **Batch**: Scheduled CSV/API updates (`BatchIngestion` + `batch_worker`)
- **Interactive Queue**: Thread-safe queue for cross-worker communication (`interaction_queue`)

Key: Streaming data gets transformed with `process_streaming_data()` to add rating based on `event_type` mapping (click→1, add_to_cart→3, purchase→5).

### Data Lake Storage Structure
Data is partitioned by: `raw/{source}/{YYYY/MM/DD}/{data_type}/{filename}_{timestamp}.{fmt}`
- **source**: 'user_behavior', 'product_api', 'user_behavior_batch'
- **data_type**: 'interactions', 'catalog', 'features'
- **Format support**: CSV, JSON, Parquet

## Critical Conventions & Patterns

### Logging Standard
**Always use `PipelineLogger` context manager**, never raw logging:
```python
from src.utils.logger import PipelineLogger

with PipelineLogger('stage_name') as logger:
    logger.info("Processing started")
    # ... code ...
    # Logger auto-tracks duration and errors on exit
```

### Configuration Pattern
Configuration is YAML-based (`config/config.yaml`). Access patterns:
- Validation rules: `config.validation.rating_min/max`, `required_columns`
- Feature definitions: `config.feature_engineering.{user|item}_features`
- Model params: `config.model.*` (n_factors, n_epochs, lr_all, reg_all, random_state)

### Feature Registration Pattern
Features must be registered in metadata via `FeatureStore.register_feature()` with:
- `feature_type`: "user", "item", or "interaction"
- `source`: reference to input data
- `transformation`: description of engineering step

Example: `feature_engineer.py` generates user/item features, then `FeatureStore.save_feature()` registers with metadata tracking.

### Model Training with MLflow
SVD matrix factorization is the standard model. MLflow uses **local file-based tracking** (SQLite):
- Tracking URI: `sqlite:///mlflow.db` (default, no server required)
- Experiment: `recomart_recommender`
- Artifacts stored in `mlruns/` directory
- To view runs: `mlflow ui` then navigate to http://127.0.0.1:5000

MLflow integration pattern:
```python
with mlflow.start_run():
    mlflow.log_param('n_factors', 50)
    mlflow.log_metric('rmse', 0.95)
    mlflow.log_artifact(model_path, artifact_path='models')
```

### Data Validation Pattern
`DataValidator` returns structured dicts, not exceptions:
```python
results = validator.validate_interactions(df)
# Returns: {'quality_score': X, 'schema': {...}, 'duplicates': {...}, ...}
```
Reports are generated as JSON in `reports/` directory. Pipeline halts if quality_score < 70.

## Project-Specific Command Workflows

### Run Full Pipeline
```bash
python main_pipeline.py
```
Executes all stages sequentially. Streaming and batch workers run in parallel threads with stop signals.

### Run Individual Components
```bash
python src/ingestion/csv_ingestion.py          # CSV data ingestion
python src/ingestion/api_ingestion.py          # Product catalog from FakeStore API
python src/validation/data_validator.py        # Data quality validation
python src/preparation/data_cleaner.py         # Data cleaning
python src/features/feature_engineer.py        # Feature generation
python src/models/recommender.py              # Model training
```

### Airflow Orchestration
```bash
airflow webserver --port 8080
airflow scheduler
```
DAG defined in `orchestration/airflow_dag.py` with XCom for inter-task communication (e.g., quality_score, counts).

### Testing
```bash
pytest tests/test_pipeline.py
```

### Development Setup
```bash
python -m venv venv
# Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## Key Dependencies & Integrations

### Core Libraries
- **Data**: pandas, numpy, scipy
- **ML**: scikit-learn (SVD, train_test_split)
- **Feature Store**: feast
- **ML Ops**: mlflow (experiment tracking), dvc (data versioning)
- **Validation**: great-expectations
- **Orchestration**: apache-airflow

### External APIs & Services
- **Product Catalog**: FakeStore API (https://fakestoreapi.com/products)
- **Message Queue**: RabbitMQ (localhost:5672, queue='user_interactions')

### Error Handling Pattern
Errors in pipeline don't raise exceptions immediately. Instead:
- Log errors with PipelineLogger (auto-logs on `__exit__`)
- Return validation/quality metrics for conditional branching
- Airflow DAG tasks catch exceptions and fail gracefully with retry logic (2 retries, 5-min delay)

## Code Organization Rules

### Module Imports
Always include: `sys.path.append(str(Path(__file__).parent.parent.parent))` to enable relative imports from root.

### Class Constructor Pattern
Ingestion/processing classes accept `storage` parameter for DataLakeStorage:
```python
class BatchIngestion:
    def __init__(self, storage: DataLakeStorage):
        self.storage = storage
```

### DataFrame Column Conventions
- **User-Item Interactions**: `user_id`, `item_id`, `rating`, `timestamp`
- **Products**: `item_id`, `name`, `category`, `price`, `description`, `image_url`
- **Streaming Events**: `user_id`, `item_id`, `event_type` (click/add_to_cart/purchase), `timestamp`

### Metadata JSON Structure
Feature metadata (`data/features/feature_metadata.json`) includes version, registration timestamp, and transformation description for reproducibility.

## When Modifying Code

1. **Adding new features**: Register via `FeatureStore.register_feature()` with proper metadata
2. **Changing validation rules**: Update `config/config.yaml` and reload in `DataValidator`
3. **Adding new data sources**: Implement new ingestion class inheriting pattern from `CSVIngestion` or `APIIngestion`
4. **Extending model**: Update hyperparameters in config, ensure MLflow logging is enabled
5. **Updating pipeline stage**: Insert new class into `main_pipeline.py` main execution flow and Airflow DAG

## Troubleshooting Patterns

- **"Quality score too low"**: Check `reports/data_quality_report.json` for missing values, duplicates, or out-of-range ratings
- **Empty DataFrame in batch**: Verify CSV file exists at `data/batch/user_interactions.csv` before running
- **RabbitMQ connection error**: Ensure RabbitMQ is running on localhost; streaming worker catches exception and retries every 2 seconds
- **MLflow tracking issues**: Check `mlruns/` directory exists and inspect `mlruns/{exp_id}/{run_id}/params` for config logging
