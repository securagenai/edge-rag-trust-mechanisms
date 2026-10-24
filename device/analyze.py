"""Summarize results: accuracy, abstention, retrieval-score gating (risk-coverage), cost."""
import json, re, string, collections, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results"
rows = [json.loads(l) for l in open(OUT / "results.jsonl")]

def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    common = collections.Counter(p) & collections.Counter(g); ns = sum(common.values())
    if ns == 0: return 0.0
    pr, rc = ns / len(p), ns / len(g); return 2 * pr * rc / (pr + rc)

for r in rows:
    ans = re.sub(r"\[\d+\]", "", r["answer"]).strip()
    r["f1"] = max((f1(ans, g) for g in r["golds"]), default=0.0)
    r["top1"] = r["scores"][0]
    r["should_answer"] = r["condition"] == "in_answerable"

def correct(r, abstain):
    if r["should_answer"]:
        return (not abstain) and r["f1"] >= 0.5
    return abstain

summary = {"n": len(rows)}
for cond in ["in_answerable", "in_unanswerable", "out_of_corpus"]:
    R = [r for r in rows if r["condition"] == cond]
    if not R: continue
    summary[cond] = {
        "n": len(R),
        "top1_score_mean": float(np.mean([r["top1"] for r in R])),
        "top1_score_median": float(np.median([r["top1"] for r in R])),
        "model_abstain_rate": float(np.mean([r["model_abstained"] for r in R])),
        "mean_f1_when_answered": float(np.mean([r["f1"] for r in R if not r["model_abstained"]] or [0])),
        "gold_retrieved_top1": float(np.mean([r["gold_retrieved_top1"] for r in R])) if cond != "out_of_corpus" else None,
    }

# risk-coverage: gate abstains if top1 < tau; combined with model's own abstention
taus = np.round(np.linspace(min(r["top1"] for r in rows) - 0.01, max(r["top1"] for r in rows), 60), 4)
curve = []
for tau in taus:
    for mode in ["gate_only", "model_only", "gate_plus_model"]:
        dec = []
        for r in rows:
            gate = r["top1"] < tau
            ab = gate if mode == "gate_only" else r["model_abstained"] if mode == "model_only" else (gate or r["model_abstained"])
            dec.append((r, ab))
        answered = [r for r, ab in dec if not ab]
        cov = len(answered) / len(rows)
        risk = (np.mean([not correct(r, False) for r in answered]) if answered else 0.0)
        acc = np.mean([correct(r, ab) for r, ab in dec])
        curve.append({"tau": float(tau), "mode": mode, "coverage": cov, "risk": float(risk), "accuracy": float(acc)})
with open(OUT / "risk_coverage.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=curve[0].keys()); w.writeheader(); w.writerows(curve)

plt.figure(figsize=(5, 4))
for mode, st in [("gate_only", "-"), ("gate_plus_model", "--")]:
    C = sorted([c for c in curve if c["mode"] == mode], key=lambda c: c["coverage"])
    plt.plot([c["coverage"] for c in C], [c["risk"] for c in C], st, label=mode.replace("_", " "))
mo = next(c for c in curve if c["mode"] == "model_only")
plt.scatter([mo["coverage"]], [mo["risk"]], marker="x", color="k", label="model self-abstention only")
plt.xlabel("Coverage (fraction answered)"); plt.ylabel("Risk (error rate among answered)")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(OUT / "risk_coverage.png", dpi=200)

plt.figure(figsize=(5, 4))
for cond in ["in_answerable", "in_unanswerable", "out_of_corpus"]:
    plt.hist([r["top1"] for r in rows if r["condition"] == cond], bins=20, alpha=.5, label=cond)
plt.xlabel("Top-1 retrieval score"); plt.ylabel("Count"); plt.legend(); plt.tight_layout()
plt.savefig(OUT / "score_distributions.png", dpi=200)

def stat(k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return {"mean": float(np.mean(v)), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95))} if v else None
cost = {k: stat(k) for k in ["retrieval_ms", "generation_ms", "prompt_tok_per_s", "gen_tok_per_s",
                             "server_rss_mb", "process_rss_mb", "mean_power_w"]}
if cost["mean_power_w"]:
    e = [r["mean_power_w"] * (r["retrieval_ms"] + r["generation_ms"]) / 1000 for r in rows if r.get("mean_power_w")]
    cost["energy_j_per_query"] = {"mean": float(np.mean(e)), "p50": float(np.median(e))}
summary["cost"] = cost
summary["system"] = json.load(open(OUT / "system_info.json"))
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
