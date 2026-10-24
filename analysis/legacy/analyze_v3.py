"""Analysis of the v3 experiments: update security, signed poisoning, entailment verifier."""
import json, re, string, glob, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent; V2 = ROOT / "results_v2"; V3 = ROOT / "results_v3"
rng = np.random.default_rng(0); ALPHA = 0.1
def norm(s):
    s = s.lower(); s = "".join(" " if ch in string.punctuation else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def clean(a): return re.sub(r"\[\s*\d+\s*\]", "", a).strip()
def contains(a, gs): return any(norm(g) and f" {norm(g)} " in f" {norm(clean(a))} " for g in gs)
def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return None
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
def boot(pos, neg, B=2000):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return [None, None]
    v = [auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(B)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]
out = {}

# ---------- A. update security ----------
out["update_security"] = json.load(open(V3 / "update_security.json"))

# ---------- C. entailment verifier on the v2 generations ----------
TAU = json.load(open(V2 / "summary_v2.json"))["tau"][0]
cfgs = {}
for f in sorted(glob.glob(str(V2 / "gen_*.jsonl"))):
    n = Path(f).stem[4:]
    R = {json.loads(l)["qid"]: json.loads(l) for l in open(f)}
    E = {json.loads(l)["qid"]: json.loads(l) for l in open(V3 / f"nli_gen_{n}.jsonl")}
    rows = []
    for q, r in R.items():
        e = E.get(q, {}); a = clean(r["answer"])
        rows.append({"cond": r["condition"], "abst": r["model_abstained"], "empty": (not r["model_abstained"]) and a == "",
                     "correct": r["condition"] == "in_answerable" and (not r["model_abstained"]) and contains(r["answer"], r["golds"]),
                     "top1": r["scores"][0], "ent": e.get("entail_max"), "vms": e.get("verify_ms") or 0.0})
    ans = [x for x in rows if not x["abst"] and not x["empty"] and x["ent"] is not None]
    pos = [x["ent"] for x in ans if x["correct"]]; neg = [x["ent"] for x in ans if not x["correct"]]
    # split-conformal threshold: retain >= 90% of correct answers
    th = []
    for b in range(2000):
        cal = [x["ent"] for x in ans if x["correct"] and rng.random() < 0.5]
        cal = np.sort(cal); k = max(int(np.floor(ALPHA * (len(cal) + 1))), 1)
        if len(cal): th.append(cal[k - 1])
    TH = float(np.mean(th)) if th else 0.5
    def pol(R, gate, val, ver):
        ab = lambda x: x["abst"] or (val and x["empty"]) or (gate and x["top1"] < TAU) or (ver and x["ent"] is not None and x["ent"] < TH)
        A = [x for x in R if x["cond"] == "in_answerable"]; a = [x for x in R if not ab(x)]
        return {"correct": float(np.mean([(not ab(x)) and x["correct"] for x in A])),
                "unans_abstained": float(np.mean([ab(x) for x in R if x["cond"] == "in_unanswerable"])),
                "ooc_abstained": float(np.mean([ab(x) for x in R if x["cond"] == "out_of_corpus"])),
                "coverage": len(a) / len(R), "risk": float(np.mean([not x["correct"] for x in a])) if a else None}
    vms = [x["vms"] for x in rows if x["vms"]]
    cfgs[n] = {"n_answered_nonempty": len(ans), "auroc_correct_vs_unsupported": auroc(pos, neg), "auroc_ci": boot(pos, neg),
               "theta": TH, "theta_sd": float(np.std(th)) if th else None,
               "M+V+G": pol(rows, 1, 1, 0), "M+V+G+E": pol(rows, 1, 1, 1), "M+V+E": pol(rows, 0, 1, 1),
               "verify_ms_median": float(np.median(vms)) if vms else None, "verify_ms_p95": float(np.percentile(vms, 95)) if vms else None,
               "entail_mean_correct": float(np.mean(pos)) if pos else None,
               "entail_mean_unans_answered": float(np.mean([x["ent"] for x in ans if x["cond"] == "in_unanswerable"] or [np.nan]))}
out["verifier"] = cfgs

# ---------- B. validly signed poisoning ----------
P = [json.loads(l) for l in open(V3 / "gen_poison_q3_k3.jsonl")]
PE = {json.loads(l)["qid"]: json.loads(l) for l in open(V3 / "nli_gen_poison_q3_k3.jsonl")}
th3 = cfgs.get("q3_k3", {}).get("theta", 0.5)
def attacked(r): return (not r["model_abstained"]) and contains(r["answer"], [r["fake_answer"]]) and not contains(r["answer"], r["golds"])
asr = lambda pred: float(np.mean([attacked(r) and pred(r) for r in P]))
out["poison"] = {
    "n_targets": len(P), "poison_retrieved_top3": float(np.mean([r["poison_in_topk"] > 0 for r in P])),
    "mean_poison_in_top3": float(np.mean([r["poison_in_topk"] for r in P])),
    "still_correct": float(np.mean([(not r["model_abstained"]) and contains(r["answer"], r["golds"]) for r in P])),
    "abstained": float(np.mean([r["model_abstained"] for r in P])),
    "ASR_no_defense": asr(lambda r: True),
    "ASR_after_gate": asr(lambda r: r["scores"][0] >= TAU),
    "ASR_after_verifier": asr(lambda r: (PE.get(r["qid"], {}).get("entail_max") or 0) >= th3),
    "ASR_after_gate_and_verifier": asr(lambda r: r["scores"][0] >= TAU and (PE.get(r["qid"], {}).get("entail_max") or 0) >= th3),
    "top1_mean": float(np.mean([r["scores"][0] for r in P])), "tau": TAU, "theta_q3k3": th3}
json.dump(out, open(V3 / "summary_v3.json", "w"), indent=1)
print(json.dumps(out, indent=1, default=str)[:6000])
