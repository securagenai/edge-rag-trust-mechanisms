"""Phase B (offline): generation for one configuration against a running llama-server.
Usage: python gen_v2.py <config_name> <k>"""
import json, sys, time, re, subprocess, threading, shutil
from pathlib import Path
import numpy as np, psutil, requests

ROOT = Path(__file__).resolve().parent; PACK = ROOT / "pack_v2"; OUT = ROOT / "results_v2"
CFG, K = sys.argv[1], int(sys.argv[2]); MAX_TOKENS = 64; SERVER = "http://127.0.0.1:8080"

class Power(threading.Thread):
    def __init__(s):
        super().__init__(daemon=True); s.samples = []; s.run_ = True
        s.ok = shutil.which("vcgencmd") is not None and s.read() is not None
    def read(s):
        try: out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=2).stdout
        except Exception: return None
        cur, volt = {}, {}
        for n, v, u in re.findall(r"(\S+)\s+\w+\(\d+\)=([\d.]+)([AV])", out):
            (cur if u == "A" else volt)[n[:-2]] = float(v)
        w = sum(cur[k] * volt[k] for k in cur if k in volt); return w if w > 0 else None
    def run(s):
        while s.run_:
            w = s.read()
            if w is not None: s.samples.append((time.time(), w))
            time.sleep(0.5)
    def mean(s, a, b):
        v = [w for t, w in s.samples if a <= t <= b]; return float(np.mean(v)) if v else None

def rss():
    for p in psutil.process_iter(["name", "memory_info"]):
        if p.info["name"] and "llama-server" in p.info["name"]: return p.info["memory_info"].rss / 2**20

corpus = json.load(open(PACK / "corpus.json"))
retr = {r["qid"]: r for r in json.load(open(OUT / "retrieval.json"))}
qs = json.load(open(ROOT / "questions_v2.json"))
f = OUT / f"gen_{CFG}.jsonl"
done = {json.loads(l)["qid"] for l in open(f)} if f.exists() else set()
pw = Power()
if pw.ok: pw.start()
with open(f, "a") as fo:
    for i, q in enumerate(qs):
        if q["qid"] in done: continue
        r = retr[q["qid"]]; ids = r["ids"][:K]
        ctx = "\n\n".join(f"[{j+1}] {corpus[p]['text']}" for j, p in enumerate(ids))
        msgs = [{"role": "system", "content":
                 "Answer the question using ONLY the numbered context passages. "
                 "Reply with a short answer phrase and the passage number in brackets. "
                 "If the context does not contain the answer, reply exactly: UNANSWERABLE"},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q['question']}"}]
        ta = time.time(); t0 = time.perf_counter()
        resp = requests.post(f"{SERVER}/v1/chat/completions", json={"messages": msgs, "max_tokens": MAX_TOKENS,
                             "temperature": 0, "seed": 42}, timeout=1800).json()
        gms = (time.perf_counter() - t0) * 1e3; tb = time.time()
        ans = resp["choices"][0]["message"]["content"].strip(); tim = resp.get("timings", {}); us = resp.get("usage", {})
        row = {"qid": q["qid"], "config": CFG, "k": K, "condition": q["condition"], "golds": q["golds"],
               "gold_pid": q["gold_pid"], "ids": ids, "scores": r["scores"], "retrieval_ms": r["retrieval_ms"],
               "answer": ans, "model_abstained": "UNANSWERABLE" in ans.upper(), "generation_ms": gms,
               "prompt_tokens": us.get("prompt_tokens"), "completion_tokens": us.get("completion_tokens"),
               "prompt_tok_per_s": tim.get("prompt_per_second"), "gen_tok_per_s": tim.get("predicted_per_second"),
               "prompt_ms": tim.get("prompt_ms"), "predicted_ms": tim.get("predicted_ms"),
               "server_rss_mb": rss(), "mean_power_w": pw.mean(ta, tb) if pw.ok else None}
        fo.write(json.dumps(row) + "\n"); fo.flush()
        print(f"[{CFG} {i+1}/{len(qs)}] {q['condition']:16s} top1={r['scores'][0]:.3f} gen={gms/1000:.1f}s -> {ans[:50]!r}", flush=True)
pw.run_ = False
try: thr = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
except Exception: thr = "n/a"
json.dump({"config": CFG, "throttled_at_end": thr}, open(OUT / f"status_{CFG}.json", "w"))
print("CONFIG DONE", CFG, thr)
