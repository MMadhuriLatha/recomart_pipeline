# recomart_pipeline
Installation:
pip install -r requirements.txt


RecoMart Recommendation System Pipeline
Project Overview
A complete end-to-end data management pipeline for a recommendation system, implementing best practices in data engineering, feature management, and ML operations.

Business Context
RecoMart is an e-commerce startup building a data-driven recommendation engine to enhance customer engagement and cross-selling. This pipeline processes user behavior data, product catalogs, and generates personalized product recommendations.

Architecture
Data Sources → Ingestion → Validation → Preparation → Feature Engineering → Feature Store → Model Training → Deployment
     ↓            ↓           ↓            ↓                ↓                   ↓              ↓
  CSV/API    Data Lake   Quality Checks  Cleaning    User/Item Features   Versioning    MLflow Tracking
Project Structure
recomart_pipeline/
├── config/
│   ├── config.yaml              # Pipeline configuration
│   └── feature_store.yaml       # Feature definitions
├── data/
│   ├── raw/                     # Raw ingested data (partitioned)
│   ├── processed/               # Cleaned data
│   ├── features/                # Engineered features
│   └── models/                  # Trained models
├── src/
│   ├── ingestion/
│   │   ├── csv_ingestion.py    # CSV data ingestion
│   │   └── api_ingestion.py    # REST API ingestion
│   ├── validation/
│   │   └── data_validator.py   # Data quality validation
│   ├── preparation/
│   │   └── data_cleaner.py     # Data cleaning & preprocessing
│   ├── features/
│   │   ├── feature_engineer.py # Feature engineering
│   │   └── feature_store.py    # Feature store implementation
│   ├── models/
│   │   └── recommender.py      # Recommendation models
│   └── utils/
│       ├── logger.py            # Logging utilities
│       └── storage.py           # Data lake storage
├── orchestration/
│   └── airflow_dag.py           # Airflow DAG
├── notebooks/
│   └── eda_analysis.ipynb       # Exploratory analysis
├── reports/
│   └── data_quality_report.json # Quality metrics
├── logs/                         # Execution logs
├── main_pipeline.py              # Main execution script
├── requirements.txt              # Dependencies
└── README.md                     # This file
Installation
Prerequisites
Python 3.8+
pip
Git
(Optional) Docker for Airflow
Setup
Clone the repository
bash
git clone <repository-url>
cd recomart_pipeline
Create virtual environment
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies
bash
pip install -r requirements.txt
Initialize directories
bash
mkdir -p data/{raw,processed,features,models,batch} logs reports
Usage
Quick Start - Run Complete Pipeline
bash
python main_pipeline.py
This executes all pipeline stages:

Data ingestion (CSV + API)
Data validation
Data cleaning
Feature engineering
Feature store population
Model training with MLflow tracking
Running Individual Components
1. Data Ingestion
CSV Ingestion:

bash
python src/ingestion/csv_ingestion.py
API Ingestion:

bash
python src/ingestion/api_ingestion.py
2. Data Validation
bash
python src/validation/data_validator.py
3. Data Preparation
bash
python src/preparation/data_cleaner.py
4. Feature Engineering
bash
python src/features/feature_engineer.py
5. Model Training
bash
python src/models/recommender.py
Using Airflow for Orchestration
Initialize Airflow
bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
Copy DAG file
bash
cp orchestration/airflow_dag.py $AIRFLOW_HOME/dags/
Start Airflow
bash
airflow webserver --port 8080
airflow scheduler
Access UI Navigate to http://localhost:8080 and trigger the recomart_recommendation_pipeline DAG
Using MLflow
Start MLflow UI
bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
Access UI Navigate to http://localhost:5000 to view experiments, metrics, and models
Pipeline Components
1. Data Ingestion
Features:

Automated periodic data fetching
Retry mechanism with exponential backoff
Error handling and logging
Partitioned storage (source/date/type)
Data Sources:

User interactions (CSV): user_id, item_id, rating, timestamp
Product catalog (REST API): product details, categories, prices
2. Data Validation
Quality Checks:

Schema validation
Missing value detection
Duplicate identification
Range validation (e.g., ratings 1-5)
Data type verification
Quality score calculation (0-100)
3. Data Preparation
Operations:

Missing value imputation
Duplicate removal
Categorical encoding
Numerical normalization
Timestamp feature extraction
4. Feature Engineering
User Features:

Interaction count
Average rating given
Rating variance/std
Days since last interaction
Rating range
Item Features:

Popularity score
Average rating received
Rating consistency
Price normalization
Category encoding
Interaction Features:

User-item rating differences
Temporal patterns
Affinity scores
5. Feature Store
Capabilities:

Feature versioning
Metadata registry
Feature retrieval for training/inference
Feature lineage tracking
API:

python
from src.features.feature_store import FeatureStore

store = FeatureStore()

# Save features
store.save_feature(df, 'user_features', version='v1')

# Retrieve features
features = store.get_feature('user_features', version='v1')

# List all features
store.list_features()
6. Model Training
Algorithm: Matrix Factorization (SVD)

Features:

Collaborative filtering
Implicit feedback handling
Cold start management
MLflow experiment tracking
Evaluation Metrics:

RMSE (Root Mean Square Error)
MAE (Mean Absolute Error)
Precision@K
Recall@K
7. Data Versioning
Using DVC:

bash
# Initialize DVC
dvc init

# Add data for tracking
dvc add data/raw data/processed data/features

# Commit changes
git add .
git commit -m "Track data versions"

# Push to remote storage
dvc push
Configuration
config.yaml
yaml
paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  features: "data/features"
  models: "data/models"

model:
  n_factors: 50
  n_epochs: 20
  learning_rate: 0.005

evaluation:
  test_size: 0.2
  metrics: ["rmse", "mae", "precision@k"]
Monitoring & Logging
Logs Location
Pipeline execution: logs/
Individual stages: logs/<stage_name>_<timestamp>.log
Log Format
2024-01-15 10:30:45 - ingestion - INFO - Ingested 10000 interactions
2024-01-15 10:31:20 - validation - INFO - Quality score: 95.5/100
API Reference
FeatureStore
python
store = FeatureStore()

# Register feature
store.register_feature(
    feature_name='user_features',
    description='User aggregated metrics',
    feature_type='user',
    source='interactions',
    transformation='groupby aggregation',
    version='v1'
)

# Create feature view
features = store.create_feature_view(
    feature_names=['user_features', 'item_features'],
    join_key='user_id',
    version='v1'
)
Recommender
python
from src.models.recommender import MatrixFactorizationRecommender

model = MatrixFactorizationRecommender(n_factors=50)
model.fit(train_data)

# Get recommendations
recommendations = model.recommend_items(
    user_id=123,
    n_recommendations=10,
    exclude_known=True
)
Performance Metrics
Sample Results:

RMSE: 0.85
MAE: 0.68
Training time: ~30 seconds (10K interactions)
Inference time: <10ms per recommendation
Deliverables Checklist
 Problem formulation document
 Data ingestion scripts (CSV + API)
 Raw data storage structure
 Data validation code
 Data quality report
 Data preparation scripts
 Feature engineering code
 Feature store implementation
 Data versioning setup
 Model training & evaluation
 MLflow tracking
 Airflow orchestration DAG
 Complete documentation
 Modular code structure
Testing
bash
# Run basic tests
python -m pytest tests/

# Test individual components
python src/ingestion/csv_ingestion.py
python src/validation/data_validator.py
Troubleshooting
Common Issues
Import errors
Ensure virtual environment is activated
Check PYTHONPATH includes project root
Missing data
Run ingestion scripts first
Check data/ directory permissions
MLflow connection errors
Verify sqlite database exists
Check tracking URI in config
Airflow DAG not showing
Verify DAG file in correct directory
Check Airflow logs for errors
Future Enhancements
 Real-time streaming ingestion with Kafka
 Advanced models (deep learning, transformers)
 A/B testing framework
 Model serving with FastAPI
 Distributed training with Spark
 Advanced monitoring with Prometheus/Grafana
 CI/CD pipeline integration
Contributing
Fork the repository
Create feature branch (git checkout -b feature/amazing-feature)
Commit changes (git commit -m 'Add amazing feature')
Push to branch (git push origin feature/amazing-feature)
Open Pull Request
License
This project is part of an academic assignment for Data Management for Machine Learning course.

Contact
For questions or issues, please open a GitHub issue or contact the development team.

Acknowledgments
Fake Store API for product data
MLflow for experiment tracking
Apache Airflow for orchestration
scikit-learn for ML utilities

 
