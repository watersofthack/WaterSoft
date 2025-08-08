# Makefile for WaterSoft Hydrological ML

.PHONY: help install install-dev test lint format clean docs

# Default target
help:
	@echo "Available targets:"
	@echo "  install      Install package and dependencies"
	@echo "  install-dev  Install package in development mode with dev dependencies"
	@echo "  test         Run tests with pytest"
	@echo "  lint         Run linting with flake8"
	@echo "  format       Format code with black and isort"
	@echo "  clean        Clean up temporary files and caches"
	@echo "  docs         Build documentation"
	@echo "  setup-env    Set up conda environment"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Environment setup
setup-env:
	conda env create -f environment.yml
	@echo "Environment created. Activate with: conda activate watersoft-ml"

# Testing
test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=src/ --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503

format:
	black src/ tests/
	isort src/ tests/

check-format:
	black --check src/ tests/
	isort --check-only src/ tests/

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Documentation
docs:
	@echo "Documentation build not implemented yet"
	@echo "Future: sphinx-build -b html docs/ docs/_build/"

# Data processing
preprocess:
	python src/preprocessing/data_extraction.py
	python src/preprocessing/attribute_processing.py

# Model training (examples)
train-lstm:
	python -m neuralhydrology.nh_run train --config-file configs/lstm_config.yml

train-transformer:
	python -m neuralhydrology.nh_run train --config-file configs/transformer_config.yml

# Development workflow
dev-setup: setup-env install-dev
	@echo "Development environment ready!"

dev-check: check-format lint test
	@echo "All development checks passed!"

# Git helpers
git-setup:
	git init
	git add .
	git commit -m "Initial commit: WaterSoft Hydrological ML project setup"
	@echo "Git repository initialized. Add remote with:"
	@echo "git remote add origin https://github.com/yourusername/LSTM-SSI.git"
	@echo "git push -u origin main"

# Quick start
quickstart: install test
	@echo "Quick start complete! Run 'make help' for more options."
