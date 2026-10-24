#!/usr/bin/env bash
# One-time setup on the Raspberry Pi (64-bit Raspberry Pi OS). Needs internet ONCE.
# Usage:  bash setup_pi.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

echo "== [1/6] System packages"
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-venv python3-dev \
     libopenblas-dev wget curl iproute2

echo "== [2/6] Build llama.cpp (takes 10-30 min on a Pi)"
COMMIT=$(python3 -c "import json;print(json.load(open('$ROOT/pins.json'))['llama_cpp_commit'])")
MARK="$ROOT/llama.cpp/.built_from_pinned_commit"
build_llama() {
  rm -rf "$ROOT/llama.cpp"
  git init -q "$ROOT/llama.cpp" && git -C "$ROOT/llama.cpp" remote add origin https://github.com/ggml-org/llama.cpp
  git -C "$ROOT/llama.cpp" fetch -q --depth 1 origin "$COMMIT" && git -C "$ROOT/llama.cpp" checkout -q FETCH_HEAD
  [ "$(git -C "$ROOT/llama.cpp" rev-parse HEAD)" = "$COMMIT" ] || { echo "ERROR: could not check out pinned llama.cpp commit"; exit 1; }
  cmake -S "$ROOT/llama.cpp" -B "$ROOT/llama.cpp/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
  cmake --build "$ROOT/llama.cpp/build" --config Release -j"$(nproc)" --target llama-server
  # marker binds the build to the pinned commit and to the SHA-256 of the resulting binary
  echo "$COMMIT $(sha256sum "$ROOT/llama.cpp/build/bin/llama-server" | cut -d' ' -f1)" > "$MARK"
}
if [ "${REPRO:-1}" = "1" ]; then
  # strict: reuse an existing binary only if this script built it from the pinned commit on a clean tree
  # and the binary still has the SHA-256 recorded at build time
  BIN="$ROOT/llama.cpp/build/bin/llama-server"
  if [ -x "$BIN" ] && [ "$(cat "$MARK" 2>/dev/null)" = "$COMMIT $(sha256sum "$BIN" 2>/dev/null | cut -d' ' -f1)" ] \
     && [ "$(git -C "$ROOT/llama.cpp" rev-parse HEAD 2>/dev/null)" = "$COMMIT" ] \
     && [ -z "$(git -C "$ROOT/llama.cpp" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    echo "llama.cpp already built from pinned commit $COMMIT"
  else
    echo "Building llama.cpp from pinned commit $COMMIT (existing build, if any, is replaced)"; build_llama
  fi
elif [ ! -x "$ROOT/llama.cpp/build/bin/llama-server" ]; then
  build_llama
fi
( cd "$ROOT/llama.cpp" && git rev-parse HEAD > "$ROOT/llama_cpp_commit.txt" )

echo "== [3/6] Python environment"
python3 -m venv "$ROOT/venv"
source "$ROOT/venv/bin/activate"
pip install --upgrade pip wheel
pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
pip install -r "$ROOT/requirements_pi.txt"   # exact versions recorded on the device
pip freeze > "$ROOT/pip_freeze.txt"

echo "== [4/6] Choose model size from RAM"
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
mkdir -p "$ROOT/models"
if [ "${REPRO:-1}" = "1" ]; then
  [ "$RAM_GB" -ge 15 ] || { echo "Strict reproduction (REPRO=1) requires a 16 GB Raspberry Pi 5; set REPRO=0 to explore"; exit 1; }
  MODEL_SIZE=3b
fi
if [ "${MODEL_SIZE:-auto}" = "auto" ]; then
  if [ "$RAM_GB" -ge 7 ]; then MODEL_SIZE=3b; else MODEL_SIZE=1.5b; fi
fi
if [ "$MODEL_SIZE" = "3b" ]; then
  URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
else
  URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
fi
FILE="$ROOT/models/$(basename "$URL")"
[ -f "$FILE" ] || wget -O "$FILE" "$URL"
if [ "${REPRO:-1}" = "1" ]; then python3 "$ROOT/verify_models.py" "$(basename "$FILE")"; fi
echo "$FILE" > "$ROOT/model_path.txt"
echo "RAM=${RAM_GB}GB -> model $MODEL_SIZE ($FILE)"

echo "== [5/6] Embedding model (cached for offline use)"
# 8 GB+: BGE-M3 (multilingual, cited in the paper). Smaller Pi: multilingual-e5-small.
if [ "$RAM_GB" -ge 7 ]; then EMB="BAAI/bge-m3"; else EMB="intfloat/multilingual-e5-small"; fi
echo "$EMB" > "$ROOT/embedding_model.txt"
python -c "from sentence_transformers import SentenceTransformer; from pins import revision; SentenceTransformer('$EMB', revision=revision('$EMB'))"

echo "== [6/6] Dataset (SQuAD 2.0 dev)"
mkdir -p "$ROOT/data"
[ -f "$ROOT/data/dev-v2.0.json" ] || wget -O "$ROOT/data/dev-v2.0.json" \
  https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

echo
echo "SETUP COMPLETE. Next:  nohup bash run_experiment.sh > run.log 2>&1 &"
