#!/usr/bin/env bash
# Full experiment. Run in the background so it survives SSH disconnects:
#   nohup bash run_experiment.sh > run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"; ROOT="$(pwd)"
source "$ROOT/venv/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MODEL="$(cat model_path.txt)"
THREADS="$(nproc)"

echo "== Building signed knowledge pack (if missing)"
[ -f pack_v1/manifest.sig ] || python build_pack.py

echo "== Running benchmark inside a network-isolated namespace (C4 conditions)"
# Only loopback exists inside: no internet, no LAN. Your SSH session is unaffected.
sudo unshare -n bash -c "
  ip link set lo up
  '$ROOT/llama.cpp/build/bin/llama-server' -m '$MODEL' -t $THREADS -c 4096 \
       --host 127.0.0.1 --port 8080 > '$ROOT/server.log' 2>&1 &
  SPID=\$!
  for i in \$(seq 1 120); do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 2; done
  runuser -u '$USER' -- env PATH='$PATH' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       '$ROOT/venv/bin/python' '$ROOT/bench.py'
  kill \$SPID
"

echo "== Analysis"
python analyze.py
echo "ALL DONE. Results in $ROOT/results"
