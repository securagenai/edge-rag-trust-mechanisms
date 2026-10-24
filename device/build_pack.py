"""Build a signed, versioned 'knowledge pack' (corpus + HNSW index + manifest).

Holds out ~10% of SQuAD articles so that some questions are out-of-corpus
(knowledge-gap queries). Also writes the question sample used by bench.py.
"""
import json, hashlib, random, time, os, sys
from pathlib import Path
import numpy as np, hnswlib
from sentence_transformers import SentenceTransformer
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

ROOT = Path(__file__).resolve().parent
SEED = 42
N_PER_CONDITION = int(os.environ.get("N_PER_CONDITION", 70))   # 3 conditions -> 210 queries
PACK = ROOT / "pack_v1"
PACK.mkdir(exist_ok=True)
rng = random.Random(SEED)

squad = json.load(open(ROOT / "data" / "dev-v2.0.json"))["data"]
titles = sorted(a["title"] for a in squad)
held_out = set(rng.sample(titles, max(1, len(titles) // 10)))

corpus, questions = [], {"in_answerable": [], "in_unanswerable": [], "out_of_corpus": []}
for art in squad:
    for para in art["paragraphs"]:
        if art["title"] in held_out:
            pid = None
        else:
            pid = len(corpus)
            corpus.append({"id": pid, "title": art["title"], "text": para["context"]})
        for qa in para["qas"]:
            golds = [a["text"] for a in qa["answers"]]
            item = {"qid": qa["id"], "question": qa["question"], "golds": golds,
                    "gold_pid": pid, "title": art["title"]}
            if art["title"] in held_out:
                if not qa["is_impossible"]:
                    questions["out_of_corpus"].append(item)
            elif qa["is_impossible"]:
                questions["in_unanswerable"].append(item)
            else:
                questions["in_answerable"].append(item)

sample = []
for cond, items in questions.items():
    for it in rng.sample(items, N_PER_CONDITION):
        it["condition"] = cond
        sample.append(it)
rng.shuffle(sample)
json.dump(sample, open(ROOT / "questions.json", "w"), indent=1)

emb_name = (ROOT / "embedding_model.txt").read_text().strip()
model = SentenceTransformer(emb_name)
prefix = "passage: " if "e5" in emb_name else ""
t0 = time.time()
X = model.encode([prefix + c["text"] for c in corpus], batch_size=16,
                 normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
embed_s = time.time() - t0

index = hnswlib.Index(space="ip", dim=X.shape[1])
index.init_index(max_elements=len(corpus), ef_construction=200, M=16, random_seed=SEED)
index.add_items(X, np.arange(len(corpus)))
index.save_index(str(PACK / "index.hnsw"))
json.dump(corpus, open(PACK / "corpus.json", "w"))

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

manifest = {
    "pack_id": "squad2-dev-knowledge-pack", "version": "1.0.0",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "embedding_model": emb_name, "dim": int(X.shape[1]), "n_passages": len(corpus),
    "held_out_titles": sorted(held_out),
    "files": {f: sha256(PACK / f) for f in ["index.hnsw", "corpus.json"]},
}
mbytes = json.dumps(manifest, sort_keys=True).encode()
(PACK / "manifest.json").write_bytes(mbytes)

keyfile = ROOT / "signing_key.pem"   # the 'publisher' key (outside the pack)
if keyfile.exists():
    sk = serialization.load_pem_private_key(keyfile.read_bytes(), password=None)
else:
    sk = Ed25519PrivateKey.generate()
    keyfile.write_bytes(sk.private_bytes(serialization.Encoding.PEM,
                        serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
(PACK / "manifest.sig").write_bytes(sk.sign(mbytes))
(ROOT / "trusted_pubkey.pem").write_bytes(sk.public_key().public_bytes(
    serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))

json.dump({"embed_seconds": embed_s, "n_passages": len(corpus),
           "n_questions": len(sample), "held_out_articles": len(held_out)},
          open(ROOT / "pack_build_stats.json", "w"), indent=1)
print(f"Pack built: {len(corpus)} passages, {len(sample)} questions, embed {embed_s:.0f}s")
