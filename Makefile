PY := ./.venv/bin/python

install:
	python3 -m venv .venv
	./.venv/bin/pip install -r requirements-dev.txt
	./scripts/fetch_jars.sh

run:
	./.venv/bin/uvicorn fingerphoto.api.app:app --host 127.0.0.1 --port 8000 --workers 1

# LAN + HTTPS so a phone on the same Wi-Fi can open /capture and use the camera
# (browsers only grant getUserMedia on HTTPS or localhost). Self-signed cert in
# certs/; accept the browser warning once on the phone. Uses 8001 so it can
# coexist with another fingerphoto on 8000.
run-phone:
	./.venv/bin/uvicorn fingerphoto.api.app:app --host 0.0.0.0 --port 8001 --workers 1 \
		--ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem

test-fast:
	$(PY) -m pytest

test:
	$(PY) -m pytest -m "not slow" && $(PY) -m pytest -m slow

docker-build:
	docker build -t fingerphoto .

docker-up:
	docker compose up -d --build

.PHONY: install run run-phone test test-fast docker-build docker-up
