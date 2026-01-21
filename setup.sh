#!/bin/bash

# RecoMart Pipeline Setup Script

echo "=========================================="
echo "RecoMart Recommendation Pipeline Setup"
echo "=========================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/{raw,processed,features,models}
mkdir -p logs
mkdir -p reports
mkdir -p notebooks
mkdir -p tests
mkdir -p config

# Initialize Git (if not already initialized)
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    echo "data/raw/*" >> .gitignore
    echo "data/processed/*" >> .gitignore
    echo "data/features/*" >> .gitignore
    echo "data/models/*" >> .gitignore
    echo "logs/*" >> .gitignore
    echo "*.pyc" >> .gitignore
    echo "__pycache__/" >> .gitignore
    echo "venv/" >> .gitignore
    echo ".ipynb_checkpoints/" >> .gitignore
    echo "mlflow.db" >> .gitignore
    echo "mlruns/" >> .gitignore
fi

# Initialize DVC
echo "Initializing DVC..."
dvc init

# Configure DVC remote (local for now)
dvc remote add -d local_storage ./dvc_storage
mkdir -p dvc_storage

# Create empty __init__.py files
echo "Creating Python package structure..."
touch src/__init__.py
touch src/ingestion/__init__.py
touch src/validation/__init__.py
touch src/preparation/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py

# Initialize MLflow
echo "Initializing MLflow..."
mlflow experiments create -n recomart_recommender 2>/dev/null || true

# Create sample config if doesn't exist
if [ ! -f "config/config.yaml" ]; then
    echo "Creating default configuration..."
    cat > config/config.yaml << EOF
project:
  name: "RecoMart Recommendation Pipeline"
  version: "1.0.0"

paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  features: "data/features"
  models: "data/models"
  logs: "logs"

validation:
  rating_min: 1
  rating_max: 5

model:
  type: "svd"
  n_factors: 50
  random_state: 42

mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "recomart_recommender"
EOF
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the complete pipeline:"
echo "   python main_pipeline.py"
echo ""
echo "3. Start MLflow UI (in separate terminal):"
echo "   mlflow ui"
echo ""
echo "4. For Airflow (optional):"
echo "   export AIRFLOW_HOME=\$(pwd)/airflow"
echo "   airflow db init"
echo "   airflow webserver -p 8080"
echo ""
echo "=========================================="