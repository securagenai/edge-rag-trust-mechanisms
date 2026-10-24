#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"; ROOT="$(pwd)"
source "$ROOT/venv/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -c "import requests; requests.get('https://example.com', timeout=3)" 2>/dev/null && echo "WARNING: network reachable" || echo "network: offline (isolated)"
echo "-- A: update security tests"; python update_security.py
echo "-- B: poisoned pack, Qwen2.5-3B k=3"
"$ROOT/llama.cpp/build/bin/llama-server" -m "$(cat models_v2_q3.txt)" -t "$(nproc)" -c 4096 -np 1 --host 127.0.0.1 --port 8080 > results_v3/server_poison.log 2>&1 &
SPID=$!
for i in $(seq 1 300); do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 2; done
python poison_gen.py || echo "poison generation failed"
kill $SPID; wait $SPID 2>/dev/null
echo "-- C: entailment verifier"; python nli_verify.py
vcgencmd get_throttled 2>/dev/null; vcgencmd measure_temp 2>/dev/null
echo "INNER V3 DONE $(date)"
