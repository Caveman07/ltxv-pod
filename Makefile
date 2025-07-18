.PHONY: help install test test-unit test-integration test-cov lint format clean run-dev run-prod docker-build docker-run docker-compose-up docker-compose-down test-local test-remote

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt || true

test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=html --cov-report=term

test-fast: ## Run fast tests (skip slow ones)
	pytest tests/ -m "not slow" -v

lint: ## Run linting
	flake8 src/ tests/ || true
	black --check src/ tests/ || true
	isort --check-only src/ tests/ || true

format: ## Format code
	black src/ tests/
	isort src/ tests/

run-dev: ## Run development server
	./scripts/start-dev.sh

run-prod: ## Run production server
	gunicorn app:app --bind 0.0.0.0:8000 --workers 4

docker-build: ## Build Docker image
	docker build -t ltxv-pod .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --gpus all ltxv-pod

docker-compose-up: ## Start with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop docker-compose
	docker-compose down

clean: ## Clean up generated files
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

test-local: ## Test local server
	LTXV_API_URL=http://localhost:8000 pytest tests/integration/ -v

test-remote: ## Test remote server (set LTXV_API_URL env var)
	pytest tests/integration/ -v 