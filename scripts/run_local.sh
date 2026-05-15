#!/bin/bash

# Script pour lancer localement l'API, MLflow et l'interface Streamlit sans Docker

# Couleurs pour le log
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Démarrage des services locaux...${NC}"

# 1. Démarrer MLflow en arrière-plan
echo -e "${GREEN}Lancement de MLflow UI sur http://localhost:5000...${NC}"
./.venv/bin/mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &
MLFLOW_PID=$!

# 2. Démarrer l'API FastAPI (Uvicorn) en arrière-plan
echo -e "${GREEN}Lancement de l'API FastAPI sur http://localhost:8000...${NC}"
export PYTHONPATH=$PYTHONPATH:$(pwd)
./.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Attendre un peu que l'API démarre
sleep 2

# 3. Démarrer Streamlit au premier plan
echo -e "${GREEN}Lancement de Streamlit UI...${NC}"
./.venv/bin/streamlit run ui/app.py --server.port 8501

# Nettoyage à la fermeture du script
trap "kill $MLFLOW_PID $API_PID; exit" SIGINT SIGTERM
