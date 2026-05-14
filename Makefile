.PHONY: install compile sync up down ingest evaluate test lint

# Compiler les .in → .txt (à relancer après chaque modif des .in)
compile:
	uv pip compile requirements.in -o requirements.txt
	uv pip compile requirements-dev.in -o requirements-dev.txt

# Installer l'environnement depuis les .txt verrouillés
install: compile
	uv pip sync requirements-dev.txt

up:
	docker-compose up -d
	@echo "✅ Stack running"
	@echo "   UI      → http://localhost:8501"
	@echo "   API     → http://localhost:8000/docs"
	@echo "   MLflow  → http://localhost:5000"

down:
	docker-compose down

ingest:
	python scripts/ingest.py --source $(SOURCE)

evaluate:
	python scripts/evaluate.py --model $(MODEL)

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ api/ scripts/
	black --check src/ api/ scripts/

pipeline:
	dvc repro
