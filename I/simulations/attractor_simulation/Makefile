# Makefile for The-Self-Remembering-Universe simulation

.PHONY: run clean env

PYTHON := python3
MAIN := plot_attractor.py
VENV := .venv

run: $(MAIN)
	@echo "Running attractor simulation..."
	$(PYTHON) $(MAIN)

env:
	@echo "Setting up virtual environment..."
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt
	@echo "✅ Virtual environment ready. Run with: source $(VENV)/bin/activate"

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ */__pycache__ *.pyc *.pyo *.log
	@echo "🧹 Done."

