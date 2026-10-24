"""Held-out (split-conformal) evaluation of the full layered policies M+V+G and M+V+G+E over 2,000 random splits.
In each split, the gate threshold (answerable queries) and the NLI threshold (correct answers) are calibrated on one half
and all metrics are computed on the other half. Usage: python heldout_eval.py [DATA_DIR]"""
import json, re, string, sys, os, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / "data"
os.chdir(DATA); rng = np.random.default_rng(3); ALPHA = 0.1; B = 2000
def norm(s):
    s = s.lower(); s = "".join(" " if ch in string.punctuation else ch for ch in s); s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def clean(a): return re.sub(r"\[\s*\d+\s*\]", "", a).strip()
def cont(a, gs): return any(norm(g) and f" {norm(g)} " in f" {norm(clean(a))} " for g in gs)
def kth(v, floor_value):
    """Split-conformal lower threshold; if the rank floor(alpha(n+1)) < 1, return a value that rejects nothing."""
    v = np.sort(v); k = int(np.floor(ALPHA * (len(v) + 1))); return float(v[k - 1]) if k >= 1 else floor_value
out = {}
for c in ["q15_k3", "q3_k1", "q3_k3", "q3_k5", "q7_k3"]:
    R = [json.loads(l) for l in open(f"results_v2/gen_{c}.jsonl")]
    E = {json.loads(l)["qid"]: json.loads(l) for l in open(f"results_v3/nli2_gen_{c}.jsonl")}
    X = []
    for r in R:
        a = clean(r["answer"])
        X.append(dict(cond=r["condition"], abst=r["model_abstained"], empty=(not r["model_abstained"]) and a == "",
                      correct=r["condition"] == "in_answerable" and (not r["model_abstained"]) and cont(r["answer"], r["golds"]),
                      top1=r["scores"][0], ent=E[r["qid"]].get("entail_max")))
    res = {"M+V+G": [], "M+V+G+E": []}
    for b in range(B):
        perm = rng.permutation(len(X)); cal = [X[i] for i in perm[: len(X) // 2]]; te = [X[i] for i in perm[len(X) // 2:]]
        tau = kth([x["top1"] for x in cal if x["cond"] == "in_answerable"], -1.0)
        th = kth([x["ent"] for x in cal if x["correct"] and x["ent"] is not None and not x["empty"]], 0.0)
        for pol, useE in [("M+V+G", False), ("M+V+G+E", True)]:
            ab = lambda x: x["abst"] or x["empty"] or x["top1"] < tau or (useE and x["ent"] is not None and x["ent"] < th)
            A = [x for x in te if x["cond"] == "in_answerable"]; ans = [x for x in te if not ab(x)]
            res[pol].append([np.mean([(not ab(x)) and x["correct"] for x in A]),
                             np.mean([ab(x) for x in te if x["cond"] == "in_unanswerable"]),
                             np.mean([ab(x) for x in te if x["cond"] == "out_of_corpus"]),
                             np.mean([not x["correct"] for x in ans]) if ans else np.nan])
    out[c] = {pol: {k: [float(np.nanmean(np.array(v)[:, i])), float(np.nanpercentile(np.array(v)[:, i], 2.5)), float(np.nanpercentile(np.array(v)[:, i], 97.5))]
                    for i, k in enumerate(["correct", "unans_abstained", "ooc_abstained", "risk"])} for pol, v in res.items()}
json.dump(out, open("results_v3/heldout_eval.json", "w"), indent=1)
for c, v in out.items():
    print(c, {p: {k: round(x[0], 3) for k, x in m.items()} for p, m in v.items()})
