"""Phase A (offline): verify + activate pack v2, retrieve top-5 for every question once.
Retrieval is model-independent, so all generation runs reuse it."""
import json, time, hashlib, shutil, requests
from pathlib import Path
import numpy as np, hnswlib
from sentence_transformers import SentenceTransformer
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

ROOT = Path(__file__).resolve().parent; PACK = ROOT / "pack_v2"
OUT = ROOT / "results_v2"; OUT.mkdir(exist_ok=True)

def verify(d):
    pub = serialization.load_pem_public_key((ROOT / "trusted_pubkey.pem").read_bytes())
    mb = (d / "manifest.json").read_bytes()
    try: pub.verify((d / "manifest.sig").read_bytes(), mb)
    except InvalidSignature: return False, "bad signature"
    man = json.loads(mb)
    for f, h in man["files"].items():
        if hashlib.sha256((d / f).read_bytes()).hexdigest() != h: return False, f"hash mismatch: {f}"
    return True, man

ts = []
for _ in range(20):
    t0 = time.perf_counter(); ok, man = verify(PACK); ts.append((time.perf_counter() - t0) * 1e3)
assert ok, man
tam = ROOT / "pack_v2_tampered"; shutil.rmtree(tam, ignore_errors=True); shutil.copytree(PACK, tam)
c = json.loads((tam / "corpus.json").read_text()); c[0]["text"] += " (tampered)"
(tam / "corpus.json").write_text(json.dumps(c)); t_ok, t_msg = verify(tam); shutil.rmtree(tam)
# signature-only tamper (manifest edited, hashes recomputed but not re-signed)
shutil.copytree(PACK, tam); m = json.loads((tam / "manifest.json").read_text()); m["version"] = "9.9.9"
(tam / "manifest.json").write_text(json.dumps(m, sort_keys=True)); s_ok, s_msg = verify(tam); shutil.rmtree(tam)

t0 = time.perf_counter()
corpus = json.load(open(PACK / "corpus.json"))
index = hnswlib.Index(space="ip", dim=man["dim"]); index.load_index(str(PACK / "index.hnsw")); index.set_ef(100)
from pins import revision
emb = SentenceTransformer(man["embedding_model"], revision=revision(man["embedding_model"])); load_ms = (time.perf_counter() - t0) * 1e3
qp = "query: " if "e5" in man["embedding_model"] else ""
try: requests.get("https://example.com", timeout=3); net = "ONLINE (not isolated!)"
except Exception: net = "offline (isolated)"

qs = json.load(open(ROOT / "questions_v2.json")); rows = []
for q in qs:
    t0 = time.perf_counter()
    v = emb.encode([qp + q["question"]], normalize_embeddings=True).astype(np.float32)
    lab, dist = index.knn_query(v, k=5); ms = (time.perf_counter() - t0) * 1e3
    rows.append({"qid": q["qid"], "ids": [int(x) for x in lab[0]], "scores": [float(1 - d) for d in dist[0]],
                 "retrieval_ms": ms})
json.dump(rows, open(OUT / "retrieval.json", "w"))
info = {"network_during_run": net, "pack_verify_ms_median": float(np.median(ts)),
        "pack_verify_ms_max": float(np.max(ts)), "pack_load_ms": load_ms,
        "tamper_corpus_rejected": not t_ok, "tamper_corpus_reason": t_msg,
        "tamper_manifest_rejected": not s_ok, "tamper_manifest_reason": s_msg,
        "n_passages": man["n_passages"], "n_questions": len(qs)}
json.dump(info, open(OUT / "pack_activation.json", "w"), indent=1); print(json.dumps(info, indent=1))
