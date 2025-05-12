# Makefile for The-Self-Remembering-Universe simulation

.PHONY: run collapse plot env clean

PYTHON := python3
MAIN := plot_attractor.py
COLLAPSE := plot_attractor.py
PLOT_SAVED := plot_saved_run.py
VENV := .venv

run: $(MAIN)
	@echo "Running attractor simulation..."
	$(PYTHON) $(MAIN)

collapse: $(COLLAPSE)
	@echo "Running collapse-triggering simulation..."
	$(PYTHON) $(COLLAPSE)

plot: $(PLOT_SAVED)
	@echo "Plotting saved simulation results..."
	$(PYTHON) $(PLOT_SAVED)

env:
	@echo "Setting up virtual environment..."
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt
	@echo "✅ Virtual environment ready. Run with: source $(VENV)/bin/activate"

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ */__pycache__ *.pyc *.pyo *.log *.npz results/*
	@echo "🧹 Done."
	
stress-test:
	@echo "Running parameter grid stress test..."
	python3 run_stress_test.py
	
summarize:
	@echo "Summarizing stress test results..."
	python3 summarize_stress_test.py
	
stress-test-extreme-1:
	@echo "Summarizing stress test results..."
	python3 run_stress_test_extreme_1.py

stress-test-extreme-2:
	@echo "Summarizing stress test results..."
	python3 run_stress_test_extreme_2.py
	
stress-test-extreme-3:
	@echo "Summarizing stress test results..."
	python3 run_stress_test_extreme_3.py

convergence_collapse_scan:
	@echo "convergence_collapse_scan..."
	python3	convergence_collapse_scan.py
