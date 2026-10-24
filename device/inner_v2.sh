#!/usr/bin/env bash
# Runs INSIDE the network namespace as the normal user.
set -uo pipefail
cd "$(dirname "$0")"; ROOT="$(pwd)"
source "$ROOT/venv/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SRV="$ROOT/llama.cpp/build/bin/llama-server"
[ -f results_v2/retrieval.json ] || python retrieve_v2.py
# config_name  model_file  k   (cheapest first, so partial results are usable)
CONFIGS="q15_k3:$(cat models_v2_q15.txt):3
q3_k3:$(cat models_v2_q3.txt):3
q3_k1:$(cat models_v2_q3.txt):1
q3_k5:$(cat models_v2_q3.txt):5
q7_k3:$(cat models_v2_q7.txt):3"
echo "$CONFIGS" | while IFS=: read -r NAME MODEL K; do
  if [ -f "results_v2/status_${NAME}.json" ]; then echo "skip $NAME (done)"; continue; fi
  if [ -z "$MODEL" ] || [ ! -s "$MODEL" ]; then echo "skip $NAME (no model file)"; continue; fi
  echo "== config $NAME  model=$(basename "$MODEL")  k=$K  $(date)"
  "$SRV" -m "$MODEL" -t "$(nproc)" -c 4096 -np 1 --host 127.0.0.1 --port 8080 > "results_v2/server_${NAME}.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 300); do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 2; done
  python gen_v2.py "$NAME" "$K" || echo "config $NAME failed"
  kill $SPID; wait $SPID 2>/dev/null; sleep 10
done
echo "INNER DONE $(date)"
