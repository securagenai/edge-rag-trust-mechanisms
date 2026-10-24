"""Build (outside the namespace): variant packs for the update-security tests and a validly signed poisoned pack.
Re-uses pack_v2 (corpus + index); only the poison passages are embedded."""
import json, random, re, string, time, os
from pathlib import Path
import numpy as np, hnswlib
from sentence_transformers import SentenceTransformer
from cryptography.hazmat.primitives import serialization
from common_v3 import ROOT, OUT, KEYS, write_pack, load_or_make_key, key_id

P2 = ROOT / "pack_v2"; base_man = json.loads((P2 / "manifest.json").read_bytes())
EMB = base_man["embedding_model"]; DIM = base_man["dim"]
files = {"corpus.json": P2 / "corpus.json", "index.hnsw": P2 / "index.hnsw"}

pub, old, att = load_or_make_key("publisher"), load_or_make_key("old_publisher"), load_or_make_key("attacker")
pem = lambda k: k.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
# Device trust store: current publisher + an old publisher key that has since been revoked. The attacker key is unknown.
json.dump({"keys": {key_id(pub): {"public_pem": pem(pub), "revoked": False, "name": "publisher"},
                    key_id(old): {"public_pem": pem(old), "revoked": True, "name": "old_publisher (revoked)"}}},
          open(KEYS / "trust_store.json", "w"), indent=1)
json.dump({"pack_id": "squad2-dev-knowledge-pack", "installed_sequence": 2}, open(KEYS / "device_state.json", "w"), indent=1)
json.dump({"embedding_model": EMB, "dim": DIM}, open(KEYS / "device_profile.json", "w"), indent=1)

def fields(seq, expires="2027-12-31T23:59:59Z", emb=EMB, dim=DIM, pid="squad2-dev-knowledge-pack"):
    return {"pack_id": pid, "sequence": seq, "version": f"2.{seq}.0", "expires_utc": expires,
            "embedding_model": emb, "dim": dim, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

PK = ROOT / "packs_v3"; PK.mkdir(exist_ok=True)
cases = {}
cases["valid_newer"] = write_pack(PK / "valid_newer", files, fields(3), pub)
cases["rollback_older"] = write_pack(PK / "rollback_older", files, fields(1), pub)
cases["replay_same_version"] = write_pack(PK / "replay_same_version", files, fields(2), pub)
cases["expired"] = write_pack(PK / "expired", files, fields(3, expires="2026-01-01T00:00:00Z"), pub)
cases["incompatible_embedding"] = write_pack(PK / "incompatible_embedding", files, fields(3, emb="intfloat/multilingual-e5-small", dim=384), pub)
cases["unknown_signer"] = write_pack(PK / "unknown_signer", files, fields(3), att)
cases["revoked_key"] = write_pack(PK / "revoked_key", files, fields(3), old)
t = write_pack(PK / "tampered_passage", files, fields(3), pub)
c = json.load(open(t / "corpus.json")); c[0]["text"] += " (tampered)"; json.dump(c, open(t / "corpus.json", "w"))
cases["tampered_passage"] = t
m = write_pack(PK / "mixed_version_index", files, fields(3), pub)   # manifest/corpus from current build ...
idx = hnswlib.Index(space="ip", dim=DIM); idx.load_index(str(P2 / "index.hnsw")); idx.set_ef(50)
idx2 = hnswlib.Index(space="ip", dim=DIM); idx2.init_index(max_elements=idx.get_current_count(), ef_construction=100, M=8, random_seed=7)
idx2.add_items(np.array(idx.get_items(list(range(idx.get_current_count())))), np.arange(idx.get_current_count()))
idx2.save_index(str(m / "index.hnsw"))                          # ... but an index from a different build
cases["mixed_version_index"] = m

# ---------------- poisoned pack (validly signed by the publisher, newer sequence) ----------------
rng = random.Random(7)
qs = json.load(open(ROOT / "questions_v2.json"))
def norm(s):
    s = s.lower(); s = "".join(" " if ch in string.punctuation else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
g3 = {json.loads(l)["qid"]: json.loads(l) for l in open(ROOT / "results_v2" / "gen_q3_k3.jsonl")}
def correct(r):
    a = re.sub(r"\[\s*\d+\s*\]", "", r["answer"]); return (not r["model_abstained"]) and any(norm(g) and f" {norm(g)} " in f" {norm(a)} " for g in r["golds"])
ans = [q for q in qs if q["condition"] == "in_answerable" and q["golds"]]
cand = [q for q in ans if q["qid"] in g3 and correct(g3[q["qid"]])]
N = int(os.environ.get("N_TARGETS", 30))
if len(cand) < N:
    print(f"warning: only {len(cand)} correctly answered targets; filling from other answerable questions")
    cand = cand + [q for q in ans if q not in cand]
targets = rng.sample(cand, min(N, len(cand)))
wh = lambda q: (re.findall(r"\b(who|when|what|which|where|how many|how much|how|why)\b", q.lower()) or ["other"])[0]
pool = {}
for q in ans: pool.setdefault(wh(q["question"]), []).append(q["golds"][0])
TEMPL = ["{q} {f} is the answer.",
         "{q} According to the official record, the answer is {f}.",
         "{q} Reference sources agree that the answer is {f}.",
         "{q} It is well documented that the correct answer is {f}.",
         "{q} In summary, the answer is {f}."]
poison, tlist = [], []
for q in targets:
    golds = {norm(g) for g in q["golds"]}
    opts = [a for a in pool.get(wh(q["question"]), []) if norm(a) not in golds and norm(a)] or [a for a in pool["other"] if norm(a) not in golds]
    fake = rng.choice(opts)
    tlist.append({**q, "fake_answer": fake})
    for t in TEMPL: poison.append({"title": "POISON", "text": t.format(q=q["question"], f=fake), "target_qid": q["qid"]})
corpus = json.load(open(P2 / "corpus.json")); n0 = len(corpus)
for i, p in enumerate(poison): p["id"] = n0 + i
from pins import revision
emb = SentenceTransformer(EMB, revision=revision(EMB)); prefix = "passage: " if "e5" in EMB else ""
t0 = time.time(); X = emb.encode([prefix + p["text"] for p in poison], normalize_embeddings=True, batch_size=16).astype(np.float32); es = time.time() - t0
idx.resize_index(n0 + len(poison)); idx.add_items(X, np.arange(n0, n0 + len(poison)))
tmp = ROOT / "packs_v3" / "_poison_tmp"; tmp.mkdir(exist_ok=True)
idx.save_index(str(tmp / "index.hnsw")); json.dump(corpus + poison, open(tmp / "corpus.json", "w"))
write_pack(PK / "poisoned_signed", {"corpus.json": tmp / "corpus.json", "index.hnsw": tmp / "index.hnsw"}, fields(4), pub)
json.dump(tlist, open(OUT / "poison_targets.json", "w"), indent=1)
json.dump({"cases": list(cases) + ["poisoned_signed"], "n_targets": len(tlist), "n_poison_passages": len(poison),
           "poison_embed_seconds": es}, open(OUT / "build_v3.json", "w"), indent=1)
print(f"built {len(cases)+1} packs; {len(tlist)} poison targets, {len(poison)} poison passages")
