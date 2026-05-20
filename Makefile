.PHONY: install test demo demo-quick gui clean

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements-ci.txt

test:
	pytest -q

demo:
	python demo_final.py --preset presentation

demo-quick:
	python demo_final.py --preset quick

gui:
	python -m pip install -r requirements-gui.txt
	python gui_dashboard.py

clean:
	rm -rf results/final_demo results/final_demo_quick results/final_demo_full .pytest_cache __pycache__ tests/__pycache__
