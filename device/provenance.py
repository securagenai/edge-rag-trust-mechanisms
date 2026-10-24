"""Record immutable identifiers of the models used: SHA-256 of GGUF files and Hugging Face snapshot revisions."""
import json, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent; out = {"gguf_sha256": {}, "hf_revisions": {}}
for f in sorted((ROOT / "models").glob("*.gguf")):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""): h.update(chunk)
    out["gguf_sha256"][f.name] = h.hexdigest(); print(f.name, h.hexdigest())
hub = Path.home() / ".cache" / "huggingface" / "hub"
for d in sorted(hub.glob("models--*")):
    ref = d / "refs" / "main"
    out["hf_revisions"][d.name.replace("models--", "").replace("--", "/")] = ref.read_text().strip() if ref.exists() else None
for name in ["nli_model.txt", "embedding_model.txt", "models_v2_q15.txt", "models_v2_q3.txt", "models_v2_q7.txt", "llama_cpp_commit.txt"]:
    p = ROOT / name
    if p.exists(): out[name] = p.read_text().strip()
json.dump(out, open(ROOT / "results_v3" / "model_provenance.json", "w"), indent=1); print(json.dumps(out["hf_revisions"], indent=1))
