#!/usr/bin/env bash
# Offline tests of the strict reproduction mode with synthetic fixtures (no real models, no hardware).
# Every scenario asserts its expected exit code and observable outcome; the script exits 1 if any assertion fails.
# Usage: bash device/tests/test_strict_mode.sh
set -u
DEV="$(cd "$(dirname "$0")/.." && pwd)"; T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
FAILS=0; PASSES=0
pass() { PASSES=$((PASSES+1)); echo "  PASS  $1"; }
fail() { FAILS=$((FAILS+1)); echo "  FAIL  $1"; }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }   # check <description> <bash condition>
run() { if "$@" > "$T/out.txt" 2>&1; then RC=0; else RC=$?; fi; }  # captures exit code without stopping the script

mkdir -p "$T/bin" && cd "$T"
for f in run_v2.sh run_v3.sh setup_pi.sh verify_models.py pins.py; do cp "$DEV/$f" .; done
printf '#!/bin/sh\nexec python3 "$@"\n' > bin/python; printf '#!/bin/sh\nexit 1\n' > bin/wget
export GITLOG="$T/gitlog.txt"
cat > bin/git <<'EOF'
#!/bin/bash
# git stub: HEAD stored in <repo>/.head; a file <repo>/.dirty makes tracked files appear modified
R=.; if [ "$1" = "-C" ]; then R="$2"; shift 2; fi
case "$1" in
  init) mkdir -p "${@: -1}"; echo init >> "$GITLOG";;
  remote) ;;
  fetch) echo "${@: -1}" > "$R/.fetched"; echo fetch >> "$GITLOG";;
  checkout) cp "$R/.fetched" "$R/.head"; echo checkout >> "$GITLOG";;
  rev-parse) cat "$R/.head" 2>/dev/null || exit 128;;
  status) [ -f "$R/.dirty" ] && echo " M CMakeLists.txt"; exit 0;;
esac
EOF
cat > bin/cmake <<'EOF'
#!/bin/bash
if [ "$1" = "--build" ]; then
  [ "${CMAKE_FAIL:-0}" = "1" ] && { echo "simulated build failure"; exit 2; }
  mkdir -p "$2/bin"; printf '#!/bin/sh\necho built\n' > "$2/bin/llama-server"; chmod +x "$2/bin/llama-server"; echo build >> "$GITLOG"
fi
exit 0
EOF
chmod +x bin/*; export PATH="$T/bin:$PATH"

# ---------- fixtures ----------
mkdir -p models
for f in qwen2.5-1.5b-instruct-q4_k_m.gguf qwen2.5-3b-instruct-q4_k_m.gguf qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf; do echo "fixture $f" > models/$f; done
python3 - <<'EOF'
import json, hashlib, os
h = {f: hashlib.sha256(open('models/' + f, 'rb').read()).hexdigest() for f in os.listdir('models')}
json.dump({"llama_cpp_commit": "aaaa1111", "hf_revisions": {"BAAI/bge-m3": "r1", "cross-encoder/nli-deberta-v3-base": "r2"},
           "gguf_sha256": h, "nli_model": "cross-encoder/nli-deberta-v3-base"}, open('pins.json', 'w'))
s = open('run_v2.sh').read(); a = s.index('M15="models/'); b = s.index('echo "== [2/4]')
open('sel.sh', 'w').write('set -euo pipefail\nROOT=$(pwd)\n' + s[a:b] + '\necho "SELECTED M7=$M7"\n')
s = open('run_v3.sh').read(); a = s.index("python - <<'PY'") + len("python - <<'PY'\n"); b = s.index("PY\npip freeze")
open('nli_block.py', 'w').write(s[a:b])
s = open('run_v3.sh').read(); a = s.index("python - <<'PY'"); b = s.index("pip freeze")
open('mini_v3.sh', 'w').write("set -euo pipefail\n" + s[a:b] + "echo CONTINUED_AFTER_NLI\n")
s = open('setup_pi.sh').read(); a = s.index('echo "== [2/6] Build llama.cpp'); b = s.index('echo "== [3/6] Python environment"')
open('llama_block.sh', 'w').write('set -euo pipefail\nROOT=$(pwd)\n' + s[a:b])
EOF
S1=models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf; ALT=models/Qwen2.5-7B-Instruct-Q4_K_M.gguf

echo "== R1: 7B selection and hash verification"
echo alternative > "$ALT"
REPRO=1 run bash sel.sh
check "R1a strict, alternative and official shards present: exit 0" '[ $RC -eq 0 ]'
check "R1a selects the official shard"                       'grep -q "SELECTED M7=$S1" out.txt'
check "R1a four hash checks OK"                              '[ "$(grep -c "^OK" out.txt)" -eq 4 ]'
mkdir -p hold && mv models/qwen2.5-7b-instruct-q4_k_m-0000* hold/
REPRO=1 run bash sel.sh
check "R1b strict, only alternative present, download fails: exit 1" '[ $RC -ne 0 ]'
check "R1b reports missing official shards"                  'grep -q "official 7B shards unavailable" out.txt'
mv hold/* models/
REPRO=0 run bash sel.sh
check "R1c exploratory: exit 0"                              '[ $RC -eq 0 ]'
check "R1c exploratory selects the alternative"              'grep -q "SELECTED M7=$ALT" out.txt'
rm "$ALT"
REPRO=1 run bash sel.sh
check "R1d strict normal path: exit 0 and official shard"    '[ $RC -eq 0 ] && grep -q "SELECTED M7=$S1" out.txt'
cp models/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf keep.gguf; echo x >> models/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf
REPRO=1 run bash sel.sh
check "R1e tampered shard: exit 1 with MISMATCH"             '[ $RC -ne 0 ] && grep -q "^MISMATCH" out.txt'
mv keep.gguf models/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf

echo "== R2: NLI checkpoint"
mkdir -p stubfail stubok
cat > stubfail/transformers.py <<'EOF'
class AutoTokenizer:
    @staticmethod
    def from_pretrained(n, revision=None): raise OSError("SIMULATED DOWNLOAD FAILURE")
class AutoModelForSequenceClassification(AutoTokenizer): pass
EOF
cat > stubok/transformers.py <<'EOF'
class _C: id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
class AutoTokenizer:
    @staticmethod
    def from_pretrained(n, revision=None): print("load", n, "rev", revision); return AutoTokenizer()
class AutoModelForSequenceClassification:
    config = _C()
    @staticmethod
    def from_pretrained(n, revision=None): return AutoModelForSequenceClassification()
EOF
echo "cross-encoder/nli-roberta-base" > nli_model.txt
REPRO=1 PYTHONPATH="stubfail:." run python3 nli_block.py
check "R2a strict, download fails: exit 1"                    '[ $RC -ne 0 ]'
check "R2a stale nli_model.txt removed"                       '[ ! -f nli_model.txt ]'
REPRO=1 run python3 -c "from pins import revision; revision('cross-encoder/nli-roberta-base')"
check "R2b strict rejects an unpinned checkpoint"             '[ $RC -ne 0 ] && grep -q "Unpinned model" out.txt'
REPRO=1 PYTHONPATH="stubok:." run python3 nli_block.py
check "R2c strict normal path: exit 0 with pinned revision"   '[ $RC -eq 0 ] && grep -q "rev r2" out.txt'
check "R2c nli_model.txt holds the pinned checkpoint"         '[ "$(cat nli_model.txt 2>/dev/null)" = "cross-encoder/nli-deberta-v3-base" ]'
REPRO=1 PYTHONPATH="stubfail:." run bash mini_v3.sh
check "R2d shell stops after NLI failure"                     '[ $RC -ne 0 ] && ! grep -q CONTINUED_AFTER_NLI out.txt'

echo "== R3: llama.cpp build"
setup_old() { rm -rf llama.cpp; mkdir -p llama.cpp/build/bin; printf '#!/bin/sh\n' > llama.cpp/build/bin/llama-server; chmod +x llama.cpp/build/bin/llama-server; echo "$1" > llama.cpp/.head; }
setup_old bbbb2222; : > "$GITLOG"
REPRO=1 run bash llama_block.sh
check "R3a strict, existing binary from another commit: exit 0" '[ $RC -eq 0 ]'
check "R3a rebuilt from the pinned commit"                    'grep -q build "$GITLOG" && [ "$(cat llama_cpp_commit.txt)" = aaaa1111 ]'
check "R3a marker records commit and binary hash"             '[ "$(cat llama.cpp/.built_from_pinned_commit)" = "aaaa1111 $(sha256sum llama.cpp/build/bin/llama-server | cut -d" " -f1)" ]'
: > "$GITLOG"; REPRO=1 run bash llama_block.sh
check "R3b re-run after pinned build reuses it (no build)"    '[ $RC -eq 0 ] && ! grep -q build "$GITLOG"'
rm llama.cpp/.built_from_pinned_commit; : > "$GITLOG"; REPRO=1 run bash llama_block.sh
check "R3c same HEAD but no marker: rebuilt"                  '[ $RC -eq 0 ] && grep -q build "$GITLOG"'
touch llama.cpp/.dirty; : > "$GITLOG"; REPRO=1 run bash llama_block.sh
check "R3d dirty tracked tree: rebuilt"                       '[ $RC -eq 0 ] && grep -q build "$GITLOG"'
printf '#!/bin/sh\necho replaced\n' > llama.cpp/build/bin/llama-server; : > "$GITLOG"; REPRO=1 run bash llama_block.sh
check "R3e binary replaced after build (hash differs): rebuilt" '[ $RC -eq 0 ] && grep -q build "$GITLOG"'
setup_old bbbb2222; : > "$GITLOG"; CMAKE_FAIL=1 REPRO=1 run bash llama_block.sh
check "R3f build failure: non-zero exit"                      '[ $RC -ne 0 ]'
check "R3f build failure: no success marker written"          '[ ! -f llama.cpp/.built_from_pinned_commit ]'
setup_old bbbb2222; : > "$GITLOG"; REPRO=0 run bash llama_block.sh
check "R3g exploratory keeps existing binary (no build)"      '[ $RC -eq 0 ] && ! grep -q build "$GITLOG" && [ "$(cat llama_cpp_commit.txt)" = bbbb2222 ]'

echo "== $PASSES passed, $FAILS failed"
[ "$FAILS" -eq 0 ]
