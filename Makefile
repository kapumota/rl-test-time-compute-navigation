.PHONY: install install-dev test coverage typecheck lint format-check quality demo demo-quick gui clean clean-results

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements-ci.txt

install-dev:
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install -e ".[dev]"

test:
	pytest

coverage:
	pytest --cov

typecheck:
	mypy

lint:
	ruff check .

format-check:
	black --check *.py tests

quality: test coverage typecheck lint format-check

demo:
	python demo_final.py --preset presentation

demo-quick:
	python demo_final.py --preset quick

gui:
	python -m pip install -r requirements-gui.txt
	python gui_dashboard.py


clean:
	find . -path "./.agente" -prune -o -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov build dist
	rm -f .coverage coverage.xml
	rm -rf *.egg-info


clean-results:
	rm -f results/*.csv results/*.json results/*.png results/*.gif results/*.mp4
