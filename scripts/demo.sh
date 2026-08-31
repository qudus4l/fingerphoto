#!/usr/bin/env bash
# Start the API and expose it over HTTPS so a phone can use the camera page.
#
#   ./scripts/demo.sh
#
# Browsers only grant getUserMedia over HTTPS or localhost, so reaching the
# capture page from a phone needs a tunnel — cloudflared's free ephemeral
# tunnel needs no account. Ctrl-C stops both processes.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
: "${FP_API_KEYS:=$(python3 -c 'import secrets; print(secrets.token_urlsafe(18))')}"
export FP_API_KEYS

command -v cloudflared >/dev/null || {
  echo "cloudflared not found. Install with: brew install cloudflared"; exit 1; }
[ -d .venv ] || { echo "No .venv — run: make install"; exit 1; }

cleanup() { kill ${SRV:-} ${TUN:-} 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "starting API on :$PORT ..."
./.venv/bin/uvicorn fingerphoto.api.app:app --host 127.0.0.1 --port "$PORT" \
  --workers 1 > /tmp/fingerphoto-api.log 2>&1 &
SRV=$!

for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null || {
  echo "API failed to start; see /tmp/fingerphoto-api.log"; tail -20 /tmp/fingerphoto-api.log; exit 1; }
echo "API up (JVM warm)."

echo "opening tunnel ..."
cloudflared tunnel --url "http://localhost:$PORT" > /tmp/fingerphoto-tunnel.log 2>&1 &
TUN=$!

URL=""
for _ in $(seq 1 45); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/fingerphoto-tunnel.log | head -1 || true)
  [ -n "$URL" ] && break
  sleep 1
done
[ -n "$URL" ] || { echo "tunnel failed; see /tmp/fingerphoto-tunnel.log"; exit 1; }

cat <<EOF

────────────────────────────────────────────────────────────
  Open on your phone:   $URL/capture
  API key:              $FP_API_KEYS

  Enter a name and that key, fill the outline with your thumb,
  and capture. Enroll 2-3 photos per person before identifying.

  Docs:   $URL/docs
  Logs:   /tmp/fingerphoto-api.log
────────────────────────────────────────────────────────────

Ctrl-C to stop.
EOF

wait $SRV
