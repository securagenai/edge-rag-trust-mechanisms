"""Pinned artefacts for strict reproduction (REPRO=1, the default). Set REPRO=0 to allow unpinned exploration."""
import json, os
from pathlib import Path
_P = json.load(open(Path(__file__).resolve().parent / "pins.json"))
STRICT = os.environ.get("REPRO", "1") == "1"
def revision(model_name):
    """Pinned Hugging Face revision. In strict mode, any model that is not pinned is rejected."""
    if not STRICT:
        return None
    if model_name not in _P["hf_revisions"]:
        raise ValueError(f"Unpinned model in strict reproduction mode: {model_name}")
    return _P["hf_revisions"][model_name]
def pinned(key):
    return _P[key]
