#!/bin/bash
set -e

URL=${1:-"http://127.0.0.1:80/predict"}
PAYLOAD='{"inputs": [[100.0, 10.0, 2.0, 4.0, 100, 1, 2, 3, 2023, 7]]}'

echo "Posting sample request to $URL"
curl -sS -X POST "$URL" -H "Content-Type: application/json" --data "$PAYLOAD" || true
