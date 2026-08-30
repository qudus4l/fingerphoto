PY := ./.venv/bin/python

install:
	python3 -m venv .venv
	./.venv/bin/pip install -r requirements-dev.txt
	./scripts/fetch_jars.sh

run:
	./.venv/bin/uvicorn fingerphoto.api.app:app --host 127.0.0.1 --port 8000 --workers 1

test-fast:
	$(PY) -m pytest

test:
	$(PY) -m pytest -m "not slow" && $(PY) -m pytest -m slow

docker-build:
	docker build -t fingerphoto .

docker-up:
	docker compose up -d --build

.PHONY: install run test test-fast docker-build docker-up
