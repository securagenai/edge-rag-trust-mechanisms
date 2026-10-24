"""Phase C (offline): post-hoc entailment verifier. For every non-abstained answer, score whether the retrieved
passages entail the statement '<question> The answer is <answer>.' with an NLI cross-encoder."""
import json, re, time, glob, numpy as np, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common_v3 import ROOT, OUT, Power
torch.set_num_threads(4)
name = (ROOT / "nli_model.txt").read_text().strip()
tok = AutoTokenizer.from_pretrained(name); model = AutoModelForSequenceClassification.from_pretrained(name).eval()
lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}; ENT = lab["entailment"]
corpus_v2 = json.load(open(ROOT / "pack_v2" / "corpus.json"))
QTEXT = {q["qid"]: q["question"] for q in json.load(open(ROOT / "questions_v2.json"))}
corpus_p = json.load(open(ROOT / "packs_v3" / "poisoned_signed" / "corpus.json"))
def clean(a): return re.sub(r"\[\s*\d+\s*\]", "", a).strip()
pw = Power(); pw.start() if pw.ok else None
jobs = sorted(glob.glob(str(ROOT / "results_v2" / "gen_*.jsonl"))) + [str(OUT / "gen_poison_q3_k3.jsonl")]
for f in jobs:
    name_out = OUT / ("nli_" + Path(f).stem + ".jsonl")
    done = {json.loads(l)["qid"] for l in open(name_out)} if name_out.exists() else set()
    corpus = corpus_p if "poison" in f else corpus_v2
    rows = [json.loads(l) for l in open(f)]
    with open(name_out, "a") as fo:
        for i, r in enumerate(rows):
            if r["qid"] in done: continue
            a = clean(r["answer"])
            if r["model_abstained"] or not a:
                fo.write(json.dumps({"qid": r["qid"], "entail_max": None, "entail": None, "verify_ms": 0.0}) + "\n"); continue
            hyp = f"{QTEXT[r['qid']]} The answer is {a}."
            prem = [corpus[p]["text"] for p in r["ids"]]
            ta = time.time(); t0 = time.perf_counter()
            with torch.no_grad():
                enc = tok(prem, [hyp] * len(prem), truncation="only_first", max_length=512, padding=True, return_tensors="pt")
                pr = torch.softmax(model(**enc).logits, dim=-1)[:, ENT].tolist()
            ms = (time.perf_counter() - t0) * 1e3
            fo.write(json.dumps({"qid": r["qid"], "entail_max": max(pr), "entail": pr, "verify_ms": ms,
                                 "mean_power_w": pw.mean(ta, time.time()) if pw.ok else None}) + "\n"); fo.flush()
            if i % 20 == 0: print(f"[nli {Path(f).stem} {i+1}/{len(rows)}] max_entail={max(pr):.3f} {ms:.0f} ms", flush=True)
    print("NLI DONE", Path(f).stem, flush=True)
pw.run_ = False
