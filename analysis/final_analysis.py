"""Final analysis for Sect. 15 (v2 experiment): metrics, Table 8 values, Figures 3-4."""
import json, re, numpy as np
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
except ImportError:
    plt = None
from pathlib import Path as _P
src = open(_P(__file__).resolve().parent / "analyze_v2.py").read(); exec(src.split("cfgs = {}")[0])
CFG = ["q15_k3", "q3_k1", "q3_k3", "q3_k5", "q7_k3"]
NAME = {"q15_k3": "1.5B, k=3", "q3_k1": "3B, k=1", "q3_k3": "3B, k=3", "q3_k5": "3B, k=5", "q7_k3": "7B, k=3"}
def isempty(r): return (not r["model_abstained"]) and re.sub(r"\[\s*\d+\s*\]", "", r["answer"]).strip() == ""
def ev(R, model, gate, val, tau=TAU):
    ab = lambda r: (gate and r["top1"] < tau) or (model and r["model_abstained"]) or (val and isempty(r))
    A = [r for r in R if r["condition"] == "in_answerable"]
    a = [r for r in R if not ab(r)]
    return {"correct": float(np.mean([(not ab(r)) and r["contains"] for r in A])),
            "unans": float(np.mean([ab(r) for r in R if r["condition"] == "in_unanswerable"])),
            "ooc": float(np.mean([ab(r) for r in R if r["condition"] == "out_of_corpus"])),
            "coverage": len(a) / len(R),
            "risk": float(np.mean([not (r["condition"] == "in_answerable" and r["contains"]) for r in a])) if a else None}
D = json.load(open("results_v2/summary_v2.json"))
out = {"tau": D["tau"], "retrieval": D["retrieval"], "activation": D["activation"], "configs": {}}
for n in CFG:
    R = [prep(json.loads(l)) for l in open(f"results_v2/gen_{n}.jsonl")]
    c = D["configs"][n]
    out["configs"][n] = {
        "correct_lenient": c["acc_answerable_lenient"], "correct_strict": c["acc_answerable_strict"],
        "empty_rate": float(np.mean([isempty(r) for r in R])),
        "policies": {p: ev(R, *f) for p, f in {"none": (0, 0, 0), "model": (1, 0, 0), "model_val": (1, 0, 1),
                     "gate_model": (1, 1, 0), "gate_model_val": (1, 1, 1)}.items()},
        "latency_median": c["latency_s"]["median"], "latency_p95": c["latency_s"]["p95"],
        "energy_median": c["energy_j_median"], "power_mean": c["power_w_mean"],
        "prefill_share": c["prefill_share"], "prompt_tokens": c["prompt_tokens_mean"],
        "prompt_tok_s": c["prompt_tok_s"], "gen_tok_s": c["gen_tok_s"],
        "rss_start_mb": R[0]["server_rss_mb"], "rss_max_mb": c["server_rss_mb_max"],
        "cv_gate_model": c["gate_plus_model_cv"], "throttled": c.get("throttled_at_end")}
json.dump(out, open("results_v2/final_summary.json", "w"), indent=1)
if plt is None:
    print("matplotlib not installed: skipping figures"); raise SystemExit(0)

# ---------------- Figures ----------------
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "pdf.fonttype": 42})
COL = {"in_answerable": "#1b6ca8", "in_unanswerable": "#e08a1e", "out_of_corpus": "#4f9a3c"}
rng2 = np.random.default_rng(1)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
for i, c in enumerate(C):
    y = S[c]; x = i + rng2.uniform(-0.18, 0.18, len(y))
    ax[0].scatter(x, y, s=6, alpha=0.55, color=COL[c], edgecolors="none")
    ax[0].boxplot(y, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color="k", lw=1),
                  boxprops=dict(lw=0.7), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
ax[0].axhline(TAU, ls="--", lw=0.8, color="k"); ax[0].text(2.45, TAU + 0.005, r"$\hat\tau$", ha="right", va="bottom")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["Answerable\n(in corpus)", "Unanswerable\n(in corpus)", "Out of\ncorpus"])
ax[0].set_ylabel("Top-1 retrieval score (cosine)"); ax[0].set_title("(a) Retrieval-score distributions (n = 100 each)", fontsize=8.5)
def roc(pos, neg):
    t = np.sort(np.unique(np.r_[pos, neg]))[::-1]
    return np.array([0] + [np.mean(neg >= x) for x in t]), np.array([0] + [np.mean(pos >= x) for x in t])
for neg, key, lab in [("out_of_corpus", "auroc_vs_ooc", "vs. out of corpus"), ("in_unanswerable", "auroc_vs_unans", "vs. unanswerable (in corpus)")]:
    f, t = roc(S["in_answerable"], S[neg]); a, lo, hi = D["retrieval"][key]
    ax[1].plot(f, t, color=COL[neg], lw=1.3, label=f"{lab}: AUROC {a:.2f} [{lo:.2f}, {hi:.2f}]")
ax[1].plot([0, 1], [0, 1], ":", color="gray", lw=0.8)
ax[1].set_xlabel("Should-abstain queries passing the gate"); ax[1].set_ylabel("Answerable queries passing the gate")
ax[1].set_title("(b) Retrieval-score gating as a classifier", fontsize=8.5)
ax[1].legend(fontsize=6.5, loc="lower right", frameon=True, framealpha=0.9, edgecolor="none")
fig.tight_layout(); fig.savefig("results_v2/Fig3.pdf"); fig.savefig("results_v2/Fig3.png", dpi=200)

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
pols = [("model", "Self-abstention", "#bdbdbd"), ("model_val", "+ output validator", "#6baed6"),
        ("gate_model_val", "+ validator + score gate", "#08519c")]
cfg_m = ["q15_k3", "q3_k3", "q7_k3"]; x = np.arange(3); w = 0.26
for j, (p, lab, col) in enumerate(pols):
    v = [out["configs"][n]["policies"][p]["risk"] for n in cfg_m]
    b = ax[0].bar(x + (j - 1) * w, v, w, color=col, label=lab)
    for xi, n in zip(x, cfg_m):
        cr = out["configs"][n]["policies"][p]["correct"]
        ax[0].text(xi + (j - 1) * w, out["configs"][n]["policies"][p]["risk"] + 0.015, f"{cr:.2f}", ha="center", fontsize=5.5, rotation=90)
ax[0].set_xticks(x); ax[0].set_xticklabels(["Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]); ax[0].set_ylim(0, 1.05)
ax[0].set_ylabel("Risk among answered queries"); ax[0].legend(fontsize=6.3, frameon=False, loc="upper right")
ax[0].set_title("(a) Layered trust mechanisms (k = 3)", fontsize=8.5)
mk = {"q15_k3": "o", "q3_k1": "s", "q3_k3": "D", "q3_k5": "^", "q7_k3": "P"}
for n in CFG:
    c = out["configs"][n]
    ax[1].scatter(c["energy_median"], c["correct_lenient"], marker=mk[n], s=36, color="#08519c" if n.startswith("q3") else "#d95f0e", zorder=3)
    dx, dy = (8, -3) if n != "q3_k5" else (8, 4)
    ax[1].annotate(f"{NAME[n]}\n{c['latency_median']:.0f} s", (c["energy_median"], c["correct_lenient"]), textcoords="offset points", xytext=(dx, dy), fontsize=6.3)
ax[1].plot([out["configs"][n]["energy_median"] for n in ["q3_k1", "q3_k3", "q3_k5"]], [out["configs"][n]["correct_lenient"] for n in ["q3_k1", "q3_k3", "q3_k5"]], ":", color="#08519c", lw=0.8)
ax[1].set_xlabel("Median energy per query (J)"); ax[1].set_ylabel("Answerable answered correctly")
ax[1].set_xlim(0, 540); ax[1].set_ylim(0, 0.9); ax[1].set_title("(b) Quality vs. energy (labels: median latency)", fontsize=8.5)
fig.tight_layout(); fig.savefig("results_v2/Fig4.pdf"); fig.savefig("results_v2/Fig4.png", dpi=200)
print(json.dumps({n: {k: (v if not isinstance(v, dict) else v) for k, v in out["configs"][n].items() if k in ["correct_lenient", "correct_strict", "empty_rate", "policies", "latency_median", "latency_p95", "energy_median", "rss_start_mb", "rss_max_mb", "prompt_tokens", "prompt_tok_s", "gen_tok_s", "prefill_share", "power_mean"]} for n in CFG}, indent=0))
