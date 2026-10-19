.PHONY: install data eda train test lint api
install:
	python -m pip install -r requirements.txt
data:
	python scripts/download_data.py
eda:
	python scripts/eda.py
train:
	python scripts/train.py
test:
	pytest -q
lint:
	ruff check .
api:
	uvicorn src.heartml.api:app --host 0.0.0.0 --port 8000
