"""Phase C, revised: post-hoc verifiers with a proper hypothesis format.
(1) NLI entailment with a rule-based question-to-declarative (QA2D) conversion of (question, answer);
(2) a deterministic lexical span check: does the normalized answer occur in a retrieved passage?"""
import json, re, string, time, glob, numpy as np, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common_v3 import ROOT, OUT, Power
torch.set_num_threads(4)
name = (ROOT / "nli_model.txt").read_text().strip()
from pins import revision
tok = AutoTokenizer.from_pretrained(name, revision=revision(name)); model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision(name)).eval()
lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}; ENT = lab["entailment"]
corpus_v2 = json.load(open(ROOT / "pack_v2" / "corpus.json"))
corpus_p = json.load(open(ROOT / "packs_v3" / "poisoned_signed" / "corpus.json"))
QTEXT = {q["qid"]: q["question"] for q in json.load(open(ROOT / "questions_v2.json"))}

AUX = r"(is|are|was|were)"
DOAUX = r"(did|does|do)"
def qa2d(q, a):
    """Rule-based conversion of a question and answer into a declarative statement."""
    q = q.strip().rstrip("?").strip(); ql = q.lower()
    m = re.match(rf"^(what|who|which)\s+{AUX}\s+(.+)$", q, re.I)
    if m: return f"{m.group(3)} {m.group(2).lower()} {a}."
    m = re.match(rf"^(what|who|which|where|when|how|why)\b(.*?)\s{DOAUX}\s+(.+)$", q, re.I)
    if m:
        wh = m.group(1).lower(); prep = "in " if wh in ("where", "when") else ""
        return f"{m.group(4)} {prep}{a}."
    m = re.match(r"^how\s+(many|much)\s+(.+)$", q, re.I)
    if m: return f"{a} {m.group(2)}."
    m = re.match(r"^(what|which)\s+(\w+)\s+(.+)$", q, re.I)
    if m and not re.match(rf"^{AUX}\b|^{DOAUX}\b", m.group(3), re.I): return f"{a} {m.group(3)}."
    m = re.match(r"^(who|what)\s+(.+)$", q, re.I)
    if m: return f"{a} {m.group(2)}."
    return f"The answer to the question \"{q}?\" is {a}."
def norm(s):
    s = s.lower(); s = "".join(" " if ch in string.punctuation else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def clean(a):
    a = re.sub(r"\[\s*\d+\s*\]", "", a); a = re.sub(r"(?i)^(answer\s*:|the answer is)\s*", "", a.strip()); return a.strip().rstrip(".")

pw = Power(); pw.start() if pw.ok else None
jobs = sorted(glob.glob(str(ROOT / "results_v2" / "gen_*.jsonl"))) + [str(OUT / "gen_poison_q3_k3.jsonl")]
for f in jobs:
    out = OUT / ("nli2_" + Path(f).stem + ".jsonl")
    done = {json.loads(l)["qid"] for l in open(out)} if out.exists() else set()
    corpus = corpus_p if "poison" in f else corpus_v2
    rows = [json.loads(l) for l in open(f)]
    with open(out, "a") as fo:
        for i, r in enumerate(rows):
            if r["qid"] in done: continue
            a = clean(r["answer"])
            if r["model_abstained"] or not a:
                fo.write(json.dumps({"qid": r["qid"], "entail_max": None, "span_support": None, "verify_ms": 0.0}) + "\n"); continue
            q = r.get("question") or QTEXT[r["qid"]]
            hyp = qa2d(q, a); prem = [corpus[p]["text"] for p in r["ids"]]
            t0 = time.perf_counter(); span = any(norm(a) and f" {norm(a)} " in f" {norm(p)} " for p in prem); sms = (time.perf_counter() - t0) * 1e3
            ta = time.time(); t0 = time.perf_counter()
            with torch.no_grad():
                enc = tok(prem, [hyp] * len(prem), truncation="only_first", max_length=512, padding=True, return_tensors="pt")
                pr = torch.softmax(model(**enc).logits, dim=-1)[:, ENT].tolist()
            ms = (time.perf_counter() - t0) * 1e3
            fo.write(json.dumps({"qid": r["qid"], "hypothesis": hyp, "entail_max": max(pr), "entail": pr, "verify_ms": ms,
                                 "span_support": span, "span_ms": sms,
                                 "mean_power_w": pw.mean(ta, time.time()) if pw.ok else None}) + "\n"); fo.flush()
            if i % 25 == 0: print(f"[nli2 {Path(f).stem} {i+1}/{len(rows)}] entail={max(pr):.3f} span={span} | {hyp[:70]}", flush=True)
    print("NLI2 DONE", Path(f).stem, flush=True)
pw.run_ = False
