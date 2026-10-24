"""Analysis of the v2 extension: model-size and top-k ablations under C4 conditions."""
import json, re, string, collections, glob
from pathlib import Path
import numpy as np
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
except ImportError:
    plt = None

import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (HERE if (HERE / "results_v2").exists() else HERE.parent / "data")
os.chdir(DATA)
ROOT = DATA; OUT = ROOT / "results_v2"
rng = np.random.default_rng(0)
C = ["in_answerable", "in_unanswerable", "out_of_corpus"]
LABEL = {"q15_k3": "Qwen2.5-1.5B, k=3", "q3_k3": "Qwen2.5-3B, k=3", "q7_k3": "Qwen2.5-7B, k=3",
         "q3_k1": "Qwen2.5-3B, k=1", "q3_k5": "Qwen2.5-3B, k=5"}

def norm(s):
    s = s.lower(); s = "".join(" " if ch in string.punctuation else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def f1(p, g):
    p, g = norm(p).split(), norm(g).split()
    c = sum((collections.Counter(p) & collections.Counter(g)).values())
    if c == 0: return 0.0
    pr, rc = c / len(p), c / len(g); return 2 * pr * rc / (pr + rc)
def prep(r):
    a = re.sub(r"\[\s*\d+\s*\]", "", r["answer"]).strip()
    r["f1"] = max([f1(a, g) for g in r["golds"]] or [0])
    r["contains"] = any(norm(g) and f" {norm(g)} " in f" {norm(a)} " for g in r["golds"])
    r["top1"] = r["scores"][0]; return r
def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
def boot(pos, neg, B=2000):
    pos, neg = np.asarray(pos), np.asarray(neg)
    v = [auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(B)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]

qs = {q["qid"]: q for q in json.load(open(ROOT / "questions_v2.json"))}
retr = json.load(open(OUT / "retrieval.json"))
for r in retr: r["condition"] = qs[r["qid"]]["condition"]; r["gold_pid"] = qs[r["qid"]]["gold_pid"]
S = {c: np.array([r["scores"][0] for r in retr if r["condition"] == c]) for c in C}
summary = {"activation": json.load(open(OUT / "pack_activation.json"))}
summary["retrieval"] = {
    "auroc_vs_ooc": [auroc(S["in_answerable"], S["out_of_corpus"])] + boot(S["in_answerable"], S["out_of_corpus"]),
    "auroc_vs_unans": [auroc(S["in_answerable"], S["in_unanswerable"])] + boot(S["in_answerable"], S["in_unanswerable"]),
    "top1_mean_sd": {c: [float(S[c].mean()), float(S[c].std(ddof=1))] for c in C},
    "recall_at": {k: float(np.mean([r["gold_pid"] in r["ids"][:k] for r in retr if r["condition"] == "in_answerable"])) for k in (1, 3, 5)},
    "retrieval_ms_median": float(np.median([r["retrieval_ms"] for r in retr]))}

# conformal threshold on answerable top-1 scores (model-independent), alpha = 0.1
ALPHA = 0.1; qids = list(qs); taus, splits = [], []
for b in range(2000):
    perm = rng.permutation(len(qids)); cal = set(qids[i] for i in perm[: len(qids) // 2])
    sc = np.sort([r["scores"][0] for r in retr if r["qid"] in cal and r["condition"] == "in_answerable"])
    k = int(np.floor(ALPHA * (len(sc) + 1))); taus.append(sc[k - 1] if k >= 1 else -1.0); splits.append(cal)  # -1 = accept all (cosine lower bound)
TAU = float(np.mean(taus)); summary["tau"] = [TAU, float(np.std(taus))]

def outcome(r, ab):
    return ((not ab) and r["contains"]) if r["condition"] == "in_answerable" else ab
def evaluate(R, tau, use_model, use_gate=True):
    ab = lambda r: (use_gate and r["top1"] < tau) or (use_model and r["model_abstained"])
    res = {"answerable_retained": float(np.mean([not ab(r) for r in R if r["condition"] == "in_answerable"])),
           "unans_abstained": float(np.mean([ab(r) for r in R if r["condition"] == "in_unanswerable"])),
           "ooc_abstained": float(np.mean([ab(r) for r in R if r["condition"] == "out_of_corpus"]))}
    answered = [r for r in R if not ab(r)]
    res["coverage"] = len(answered) / len(R)
    res["risk"] = float(np.mean([not outcome(r, False) for r in answered])) if answered else None
    res["accuracy"] = float(np.mean([outcome(r, ab(r)) for r in R]))
    return res
def cv(R, use_model):
    vals = collections.defaultdict(list)
    for tau, cal in zip(taus[:500], splits[:500]):
        T = [r for r in R if r["qid"] not in cal]
        for k, v in evaluate(T, tau, use_model).items():
            if v is not None: vals[k].append(v)
    return {k: [float(np.mean(v)), float(np.std(v))] for k, v in vals.items()}

cfgs = {}
for f in sorted(glob.glob(str(OUT / "gen_*.jsonl"))):
    name = Path(f).stem[4:]; R = [prep(json.loads(l)) for l in open(f)]
    if len(R) < len(qs): print(f"warning: {name} incomplete ({len(R)}/{len(qs)})")
    A = [r for r in R if r["condition"] == "in_answerable"]
    pre = np.array([r["prompt_ms"] or 0 for r in R]); dec = np.array([r["predicted_ms"] or 0 for r in R])
    lat = np.array([(r["retrieval_ms"] + r["generation_ms"]) / 1000 for r in R])
    pw = [r["mean_power_w"] for r in R if r["mean_power_w"]]
    en = [r["mean_power_w"] * (r["retrieval_ms"] + r["generation_ms"]) / 1000 for r in R if r["mean_power_w"]]
    cfgs[name] = {
        "n": len(R), "label": LABEL.get(name, name),
        "no_abstention": evaluate(R, -1, False, False),
        "model_only": evaluate(R, -1, True, False),
        "gate_only": evaluate(R, TAU, False),
        "gate_plus_model": evaluate(R, TAU, True),
        "gate_plus_model_cv": cv(R, True), "model_only_note": "model-only is threshold-free",
        "acc_answerable_lenient": float(np.mean([(not r["model_abstained"]) and r["contains"] for r in A])),
        "acc_answerable_strict": float(np.mean([(not r["model_abstained"]) and r["f1"] >= 0.5 for r in A])),
        "precision_when_answered_lenient": float(np.mean([r["contains"] for r in A if not r["model_abstained"]] or [np.nan])),
        "latency_s": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "mean": float(lat.mean())},
        "prefill_share": float(np.mean(pre / np.maximum(pre + dec, 1e-9))),
        "prompt_tokens_mean": float(np.mean([r["prompt_tokens"] for r in R])),
        "prompt_tok_s": float(np.median([r["prompt_tok_per_s"] for r in R])),
        "gen_tok_s": float(np.median([r["gen_tok_per_s"] for r in R])),
        "power_w_mean": float(np.mean(pw)) if pw else None,
        "energy_j_median": float(np.median(en)) if en else None,
        "server_rss_mb_max": float(max(r["server_rss_mb"] or 0 for r in R)),
    }
    st = OUT / f"status_{name}.json"
    if st.exists(): cfgs[name]["throttled_at_end"] = json.load(open(st))["throttled_at_end"]
    gain = {k: cfgs[name]["gate_plus_model"][k] - cfgs[name]["model_only"][k]
            for k in ["answerable_retained", "unans_abstained", "ooc_abstained", "risk"]}
    cfgs[name]["gate_added_value"] = gain
summary["configs"] = cfgs
json.dump(summary, open(OUT / "summary_v2.json", "w"), indent=1)

# ------------------ tables ------------------
lines = ["config,answerable_retained_model,answerable_retained_combined,unans_abstained_model,unans_abstained_combined,"
         "ooc_abstained_model,ooc_abstained_combined,risk_model,risk_combined,acc_lenient,acc_strict,latency_median_s,"
         "latency_p95_s,prefill_share,energy_J_median,rss_max_MB"]
for n, c in cfgs.items():
    m, g = c["model_only"], c["gate_plus_model"]
    lines.append(",".join(str(round(x, 3)) if isinstance(x, float) else str(x) for x in [
        n, m["answerable_retained"], g["answerable_retained"], m["unans_abstained"], g["unans_abstained"],
        m["ooc_abstained"], g["ooc_abstained"], m["risk"], g["risk"], c["acc_answerable_lenient"],
        c["acc_answerable_strict"], c["latency_s"]["median"], c["latency_s"]["p95"], c["prefill_share"],
        c["energy_j_median"] or 0, c["server_rss_mb_max"]]))
(OUT / "table_v2.csv").write_text("\n".join(lines) + "\n")

# ------------------ figures ------------------
if plt is None:
    print("matplotlib not installed: skipping figures"); raise SystemExit(0)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "pdf.fonttype": 42})
models = [n for n in ["q15_k3", "q3_k3", "q7_k3"] if n in cfgs]
if models:
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    x = np.arange(3); w = 0.8 / (2 * len(models))
    shades = ["#9ecae1", "#4292c6", "#08519c"]
    for i, n in enumerate(models):
        for j, pol in enumerate(["model_only", "gate_plus_model"]):
            e = cfgs[n][pol]; v = [e["answerable_retained"], e["unans_abstained"], e["ooc_abstained"]]
            pos = x + (i * 2 + j - len(models) + 0.5) * w
            ax[0].bar(pos, v, w, color=shades[i], hatch=None if j == 0 else "///", edgecolor="white", lw=0.3,
                      label=f"{cfgs[n]['label'].split(',')[0]} – {'model' if j == 0 else 'gate+model'}")
    ax[0].set_xticks(x); ax[0].set_xticklabels(["Answerable\nretained", "Unanswerable\nabstained", "Out-of-corpus\nabstained"])
    ax[0].set_ylim(0, 1.35); ax[0].set_yticks(np.linspace(0, 1, 6)); ax[0].set_ylabel("Rate")
    ax[0].legend(fontsize=5.6, ncol=2, frameon=False, loc="upper center"); ax[0].set_title("(a) Self-abstention vs. gate + model, by model size", fontsize=8.5)
    taus_s = np.linspace(min(r["scores"][0] for r in retr) - 0.01, max(r["scores"][0] for r in retr), 150)
    for i, n in enumerate(models):
        R = [prep(json.loads(l)) for l in open(OUT / f"gen_{n}.jsonl")]
        pts = [evaluate(R, t, True) for t in taus_s]; pts = [p for p in pts if p["risk"] is not None]
        ax[1].plot([p["coverage"] for p in pts], [p["risk"] for p in pts], color=shades[i], lw=1.2, label=cfgs[n]["label"].split(",")[0] + " (gate+model sweep)")
        mo = cfgs[n]["model_only"]; ax[1].scatter([mo["coverage"]], [mo["risk"]], color=shades[i], marker="x", s=22)
    ax[1].set_xlabel("Coverage"); ax[1].set_ylabel("Risk among answered queries"); ax[1].set_xlim(0, 1.02)
    ax[1].legend(fontsize=6, frameon=False, loc="lower right"); ax[1].set_title("(b) Risk–coverage by model size", fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "explore_model_size.pdf"); fig.savefig(OUT / "explore_model_size.png", dpi=200)
ks = [n for n in ["q3_k1", "q3_k3", "q3_k5"] if n in cfgs]
if ks:
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5)); kv = [cfgs[n] for n in ks]; kk = [int(n[-1]) for n in ks]
    ax[0].plot(kk, [c["latency_s"]["median"] for c in kv], "o-", color="#08519c", label="Median latency (s)")
    ax[0].set_xlabel("Retrieved passages in prompt (k)"); ax[0].set_ylabel("Median latency (s)", color="#08519c"); ax[0].set_xticks(kk)
    if all(c["energy_j_median"] for c in kv):
        a2 = ax[0].twinx(); a2.plot(kk, [c["energy_j_median"] for c in kv], "s--", color="#d95f0e"); a2.set_ylabel("Median energy (J)", color="#d95f0e")
    ax[0].set_title("(a) Cost vs. context size", fontsize=8.5)
    ax[1].plot(kk, [c["acc_answerable_lenient"] for c in kv], "o-", color="#08519c", label="Answerable correct (lenient)")
    ax[1].plot(kk, [summary["retrieval"]["recall_at"][str(k)] if str(k) in summary["retrieval"]["recall_at"] else summary["retrieval"]["recall_at"][k] for k in kk], "^:", color="gray", label="Gold passage in top-k")
    ax[1].plot(kk, [c["model_only"]["unans_abstained"] for c in kv], "s--", color="#d95f0e", label="Unanswerable abstained (model)")
    ax[1].set_xlabel("Retrieved passages in prompt (k)"); ax[1].set_ylabel("Rate"); ax[1].set_xticks(kk); ax[1].set_ylim(0, 1)
    ax[1].legend(fontsize=6.3, frameon=False, loc="lower right"); ax[1].set_title("(b) Quality vs. context size (Qwen2.5-3B)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "explore_context_size.pdf"); fig.savefig(OUT / "explore_context_size.png", dpi=200)
print(json.dumps({"tau": summary["tau"], "retrieval": summary["retrieval"],
                  "configs": {n: {"model_only": c["model_only"], "gate_plus_model": c["gate_plus_model"],
                                  "gate_added_value": c["gate_added_value"], "acc_lenient": c["acc_answerable_lenient"],
                                  "latency": c["latency_s"], "energy_J": c["energy_j_median"]} for n, c in cfgs.items()}}, indent=1))
