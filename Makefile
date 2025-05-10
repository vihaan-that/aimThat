# Makefile for NoScope9000-ML-Analysis

.PHONY: setup test lint clean visualize predict

# Setup development environment
setup:
	pip install -r requirements.txt

# Run tests
test:
	python -m unittest discover tests

# Generate visualizations
visualize:
	python src/generate_plots.py

# Run a prediction example
predict:
	python src/predict.py --model ensemble --distance 20.5 --elevation 1.5 --tiltx 70.3 --tilty 150.4 --xdiff 10.2 --ydiff 1.4 --zdiff 18.3

# Run linting 
lint:
	flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Clean build artifacts
clean:
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Create distribution package
dist:
	python setup.py sdist bdist_wheel

# Install in development mode
dev-install:
	pip install -e .

# Show help
help:
	@echo "Available commands:"
	@echo "  make setup        - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make visualize    - Generate visualizations"
	@echo "  make predict      - Run prediction example"
	@echo "  make lint         - Check code style with flake8"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make dist         - Create distribution package"
	@echo "  make dev-install  - Install package in development mode"
