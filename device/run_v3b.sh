#!/usr/bin/env bash
# Re-run only the verifiers with the corrected hypothesis format (~30-40 min). Fully offline.
#   setsid nohup bash run_v3b.sh > run_v3b.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"; ROOT="$(pwd)"
source "$ROOT/venv/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
sudo unshare -n bash -c "ip link set lo up; runuser -u '$USER' -- bash -c 'source $ROOT/venv/bin/activate; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; cd $ROOT; python nli_verify_v2.py; vcgencmd get_throttled; vcgencmd measure_temp'"
echo "ALL DONE V3B  $(date)"
