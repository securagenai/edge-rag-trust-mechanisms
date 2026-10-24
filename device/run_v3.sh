#!/usr/bin/env bash
# v3 experiments: update security, validly signed poisoning, entailment verifier (~1-1.5 h on a Pi 5).
#   setsid nohup bash run_v3.sh > run_v3.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")"; ROOT="$(pwd)"
source "$ROOT/venv/bin/activate"
echo "== [1/4] NLI model download (needs internet)  $(date)"
pip install -q sentencepiece protobuf
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from pathlib import Path
from pins import revision, pinned, STRICT
Path("nli_model.txt").unlink(missing_ok=True)          # never reuse a model name from an earlier run
names = [pinned("nli_model")] + ([] if STRICT else ["cross-encoder/nli-roberta-base"])
for name in names:
    try:
        AutoTokenizer.from_pretrained(name, revision=revision(name)); m = AutoModelForSequenceClassification.from_pretrained(name, revision=revision(name))
        assert "entailment" in [v.lower() for v in m.config.id2label.values()]
    except Exception as e:
        print("failed", name, e); continue
    Path("nli_model.txt").write_text(name); print("NLI model:", name); break
else:
    raise SystemExit("ERROR: no NLI checkpoint could be loaded" + (" (strict reproduction)" if STRICT else ""))
PY
pip freeze > pip_freeze_v3.txt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "== [2/4] Build variant packs and signed poisoned pack  $(date)"
[ -f results_v3/build_v3.json ] || python build_v3.py
echo "== [3/4] Offline experiments in network-isolated namespace  $(date)"
sudo unshare -n bash -c "ip link set lo up; runuser -u '$USER' -- bash '$ROOT/inner_v3.sh'"
echo "ALL DONE V3  $(date)"
