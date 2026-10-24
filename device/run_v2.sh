#!/usr/bin/env bash
# Extension experiment (one command, runs overnight, ~12-14 h on a Pi 5):
#   setsid nohup bash run_v2.sh > run_v2.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")"; ROOT="$(pwd)"
source "$ROOT/venv/bin/activate"
echo "== [1/4] Downloading models (needs internet)  $(date)"
mkdir -p models
M15="models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
[ -s "$M15" ] || wget -c --progress=dot:giga -O "$M15" \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
M3="models/qwen2.5-3b-instruct-q4_k_m.gguf"
[ -s "$M3" ] || wget -c --progress=dot:giga -O "$M3" \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
get() { wget -c --progress=dot:giga -O "$2" "$1" || { rm -f "$2"; return 1; }; }
S1="models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"; S2="models/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
B7="models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"; M7=""
if [ "${REPRO:-1}" = "0" ] && [ -s "$B7" ]; then M7="$B7"   # exploration only
elif [ -s "$S1" ] && [ -s "$S2" ]; then M7="$S1"
elif get "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf" "$S1" && \
     get "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf" "$S2"; then M7="$S1"
elif [ "${REPRO:-1}" = "0" ] && get "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf" "$B7"; then M7="$B7"  # exploration only
elif [ "${REPRO:-1}" = "1" ]; then echo "ERROR: official 7B shards unavailable; aborting strict reproduction"; exit 1
else echo "WARNING: 7B download failed; the q7 configuration will be skipped"; fi
echo "$ROOT/$M15" > models_v2_q15.txt; echo "$ROOT/$M3" > models_v2_q3.txt; if [ -n "$M7" ]; then echo "$ROOT/$M7" > models_v2_q7.txt; else : > models_v2_q7.txt; fi
ls -la models
if [ "${REPRO:-1}" = "1" ]; then
  # the files actually selected must be exactly the pinned, hash-verified artefacts
  [ "$M7" = "$S1" ] && [ -s "$S2" ] || { echo "Selected 7B model is not the pinned official shard pair: aborting"; exit 1; }
  python verify_models.py "$(basename "$M15")" "$(basename "$M3")" "$(basename "$S1")" "$(basename "$S2")" \
    || { echo "Model hash mismatch: aborting strict reproduction"; exit 1; }
fi

echo "== [2/4] Building knowledge pack v2 (~30 min)  $(date)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -f pack_v2/manifest.sig ] || python build_pack_v2.py

echo "== [3/4] Offline benchmark in network-isolated namespace  $(date)"
sudo unshare -n bash -c "ip link set lo up; runuser -u '$USER' -- bash '$ROOT/inner_v2.sh'"

echo "== [4/4] Analysis  $(date)"
python analyze_v2.py || echo "analysis failed (results are still saved)"
echo "ALL DONE V2  $(date)"
