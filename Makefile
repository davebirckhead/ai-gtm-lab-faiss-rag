setup:
	uv venv && uv pip install -r requirements.txt
lint:
	ruff .
format:
	black .
test:
	pytest
eval:
	python apps/track-a-b2c-product-advisor/eval.py
