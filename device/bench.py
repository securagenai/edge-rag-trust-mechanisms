"""Run the offline (network-isolated) benchmark on the Pi.

1. Activates the knowledge pack: verifies Ed25519 signature + SHA-256 hashes (timed),
   and runs a tamper test on a copy (expects rejection).
2. For each question: retrieves top-k passages, records scores, asks the local LLM
   (llama-server on localhost) to answer ONLY from the context or say UNANSWERABLE.
3. Logs latency, tokens/s, memory, and (Pi 5) approximate power.
"""
import json, time, hashlib, shutil, subprocess, threading, re, os, platform
from pathlib import Path
import numpy as np, hnswlib, psutil, requests
from sentence_transformers import SentenceTransformer
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

ROOT = Path(__file__).resolve().parent
PACK = ROOT / "pack_v1"
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
TOPK, MAX_TOKENS = 3, 64
SERVER = "http://127.0.0.1:8080"

def verify_pack(pack_dir):
    pub = serialization.load_pem_public_key((ROOT / "trusted_pubkey.pem").read_bytes())
    mbytes = (pack_dir / "manifest.json").read_bytes()
    try:
        pub.verify((pack_dir / "manifest.sig").read_bytes(), mbytes)
    except InvalidSignature:
        return False, "bad signature"
    man = json.loads(mbytes)
    for f, h in man["files"].items():
        if hashlib.sha256((pack_dir / f).read_bytes()).hexdigest() != h:
            return False, f"hash mismatch: {f}"
    return True, man

# ---------- power sampling (Raspberry Pi 5 PMIC; approximate) ----------
class PowerSampler(threading.Thread):
    def __init__(self, period=0.5):
        super().__init__(daemon=True); self.period = period
        self.samples = []; self.running = True
        self.available = shutil.which("vcgencmd") is not None and self.read() is not None
    def read(self):
        try:
            out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True,
                                 text=True, timeout=2).stdout
        except Exception:
            return None
        cur, volt = {}, {}
        for name, val, unit in re.findall(r"(\S+)\s+\w+\(\d+\)=([\d.]+)([AV])", out):
            base = name[:-2]
            (cur if unit == "A" else volt)[base] = float(val)
        w = sum(cur[k] * volt[k] for k in cur if k in volt)
        return w if w > 0 else None
    def run(self):
        while self.running:
            w = self.read()
            if w is not None: self.samples.append((time.time(), w))
            time.sleep(self.period)
    def mean_between(self, t0, t1):
        v = [w for t, w in self.samples if t0 <= t <= t1]
        return float(np.mean(v)) if v else None

def server_rss_mb():
    for p in psutil.process_iter(["name", "memory_info"]):
        if p.info["name"] and "llama-server" in p.info["name"]:
            return p.info["memory_info"].rss / 2**20
    return None

def main():
    # ---- pack activation (C3/C4 controlled update) ----
    t0 = time.perf_counter(); ok, man = verify_pack(PACK); verify_ms = (time.perf_counter() - t0) * 1e3
    assert ok, f"pack rejected: {man}"
    tam = ROOT / "pack_tampered"; shutil.rmtree(tam, ignore_errors=True); shutil.copytree(PACK, tam)
    c = json.loads((tam / "corpus.json").read_text()); c[0]["text"] += " (tampered)"
    (tam / "corpus.json").write_text(json.dumps(c))
    tamper_ok, tamper_msg = verify_pack(tam); shutil.rmtree(tam)

    t0 = time.perf_counter()
    corpus = json.load(open(PACK / "corpus.json"))
    index = hnswlib.Index(space="ip", dim=man["dim"])
    index.load_index(str(PACK / "index.hnsw")); index.set_ef(100)
    emb_name = man["embedding_model"]
    embedder = SentenceTransformer(emb_name)
    load_ms = (time.perf_counter() - t0) * 1e3
    qprefix = "query: " if "e5" in emb_name else ""

    # network isolation check (should fail inside the isolated namespace)
    try:
        requests.get("https://example.com", timeout=3); net = "ONLINE (not isolated!)"
    except Exception:
        net = "offline (isolated)"

    sysinfo = {"platform": platform.platform(), "machine": platform.machine(),
               "cpu_count": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
               "model_file": Path((ROOT / "model_path.txt").read_text().strip()).name,
               "embedding_model": emb_name, "network_during_run": net,
               "pack_verify_ms": verify_ms, "pack_load_ms": load_ms,
               "tamper_test_rejected": (not tamper_ok), "tamper_reason": tamper_msg}
    json.dump(sysinfo, open(OUT / "system_info.json", "w"), indent=1)
    print(json.dumps(sysinfo, indent=1))

    power = PowerSampler(); 
    if power.available: power.start()
    questions = json.load(open(ROOT / "questions.json"))
    done = set()
    resf = OUT / "results.jsonl"
    if resf.exists():
        done = {json.loads(l)["qid"] for l in open(resf)}
    with open(resf, "a") as fout:
        for i, q in enumerate(questions):
            if q["qid"] in done: continue
            t_start = time.time()
            t0 = time.perf_counter()
            qv = embedder.encode([qprefix + q["question"]], normalize_embeddings=True).astype(np.float32)
            labels, dists = index.knn_query(qv, k=TOPK)
            retr_ms = (time.perf_counter() - t0) * 1e3
            ids = [int(x) for x in labels[0]]
            scores = [float(1 - d) for d in dists[0]]          # hnswlib 'ip' distance = 1 - dot
            ctx = "\n\n".join(f"[{j+1}] {corpus[pid]['text']}" for j, pid in enumerate(ids))
            msgs = [{"role": "system", "content":
                     "Answer the question using ONLY the numbered context passages. "
                     "Reply with a short answer phrase and the passage number in brackets. "
                     "If the context does not contain the answer, reply exactly: UNANSWERABLE"},
                    {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q['question']}"}]
            t0 = time.perf_counter()
            r = requests.post(f"{SERVER}/v1/chat/completions", json={
                "messages": msgs, "max_tokens": MAX_TOKENS, "temperature": 0, "seed": 42},
                timeout=900).json()
            gen_ms = (time.perf_counter() - t0) * 1e3
            t_end = time.time()
            answer = r["choices"][0]["message"]["content"].strip()
            tim = r.get("timings", {})
            row = {**{k: q[k] for k in ["qid", "condition", "question", "golds", "gold_pid"]},
                   "retrieved_ids": ids, "scores": scores,
                   "gold_retrieved_top1": q["gold_pid"] is not None and ids[0] == q["gold_pid"],
                   "answer": answer, "model_abstained": "UNANSWERABLE" in answer.upper(),
                   "retrieval_ms": retr_ms, "generation_ms": gen_ms,
                   "prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                   "completion_tokens": r.get("usage", {}).get("completion_tokens"),
                   "prompt_tok_per_s": tim.get("prompt_per_second"),
                   "gen_tok_per_s": tim.get("predicted_per_second"),
                   "server_rss_mb": server_rss_mb(),
                   "process_rss_mb": psutil.Process().memory_info().rss / 2**20,
                   "mean_power_w": power.mean_between(t_start, t_end) if power.available else None}
            fout.write(json.dumps(row) + "\n"); fout.flush()
            print(f"[{i+1}/{len(questions)}] {q['condition']:16s} top1={scores[0]:.3f} "
                  f"gen={gen_ms/1000:.1f}s -> {answer[:60]!r}")
    power.running = False
    try:
        thr = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
        tmp = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True).stdout.strip()
    except Exception:
        thr = tmp = "n/a"
    info = json.load(open(OUT / "system_info.json"))
    info.update({"throttled_at_end": thr, "temp_at_end": tmp, "power_sampling": power.available})
    json.dump(info, open(OUT / "system_info.json", "w"), indent=1)
    print("BENCHMARK COMPLETE", thr, tmp)

if __name__ == "__main__":
    main()
