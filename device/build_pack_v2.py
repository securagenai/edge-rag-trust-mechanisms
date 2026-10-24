"""v2 knowledge pack: holds out ~20% of SQuAD 2.0 dev articles (more diverse
out-of-corpus queries) and samples N_PER_CONDITION questions per condition."""
import json, hashlib, random, time, os
from pathlib import Path
import numpy as np, hnswlib
from sentence_transformers import SentenceTransformer
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

ROOT = Path(__file__).resolve().parent
SEED = 2026
N = int(os.environ.get("N_PER_CONDITION", 100))
HOLDOUT = float(os.environ.get("HOLDOUT_FRAC", 0.2))
PACK = ROOT / "pack_v2"; PACK.mkdir(exist_ok=True)
rng = random.Random(SEED)

squad = json.load(open(ROOT / "data" / "dev-v2.0.json"))["data"]
titles = sorted(a["title"] for a in squad)
held_out = set(rng.sample(titles, max(1, round(len(titles) * HOLDOUT))))

corpus, qs = [], {"in_answerable": [], "in_unanswerable": [], "out_of_corpus": []}
for art in squad:
    for para in art["paragraphs"]:
        pid = None
        if art["title"] not in held_out:
            pid = len(corpus); corpus.append({"id": pid, "title": art["title"], "text": para["context"]})
        for qa in para["qas"]:
            it = {"qid": qa["id"], "question": qa["question"], "golds": [a["text"] for a in qa["answers"]],
                  "gold_pid": pid, "title": art["title"]}
            if art["title"] in held_out:
                if not qa["is_impossible"]: qs["out_of_corpus"].append(it)
            elif qa["is_impossible"]: qs["in_unanswerable"].append(it)
            else: qs["in_answerable"].append(it)

sample = []
for cond, items in qs.items():
    for it in rng.sample(items, N):
        it["condition"] = cond; sample.append(it)
rng.shuffle(sample)
json.dump(sample, open(ROOT / "questions_v2.json", "w"), indent=1)

emb_name = (ROOT / "embedding_model.txt").read_text().strip()
from pins import revision
model = SentenceTransformer(emb_name, revision=revision(emb_name))
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

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest = {"pack_id": "squad2-dev-knowledge-pack", "version": "2.0.0",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "embedding_model": emb_name, "dim": int(X.shape[1]), "n_passages": len(corpus),
            "held_out_titles": sorted(held_out),
            "files": {f: sha(PACK / f) for f in ["index.hnsw", "corpus.json"]}}
mbytes = json.dumps(manifest, sort_keys=True).encode()
(PACK / "manifest.json").write_bytes(mbytes)
keyfile = ROOT / "signing_key.pem"
if keyfile.exists():
    sk = serialization.load_pem_private_key(keyfile.read_bytes(), password=None)
else:
    sk = Ed25519PrivateKey.generate()
    keyfile.write_bytes(sk.private_bytes(serialization.Encoding.PEM,
                        serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
(PACK / "manifest.sig").write_bytes(sk.sign(mbytes))
(ROOT / "trusted_pubkey.pem").write_bytes(sk.public_key().public_bytes(
    serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
json.dump({"embed_seconds": embed_s, "n_passages": len(corpus), "n_questions": len(sample),
           "held_out_articles": len(held_out), "n_articles": len(titles)},
          open(ROOT / "pack_v2_build_stats.json", "w"), indent=1)
print(f"Pack v2: {len(corpus)} passages, {len(held_out)}/{len(titles)} articles held out, "
      f"{len(sample)} questions, embed {embed_s:.0f}s")
