import json, re, string, glob
from pathlib import Path
import numpy as np
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (HERE if (HERE / "results_v2").exists() else HERE.parent / "data")
os.chdir(DATA)
ROOT=Path('.'); V2=ROOT/'results_v2'; V3=ROOT/'results_v3'; rng=np.random.default_rng(0); ALPHA=0.1
def norm(s):
    s=s.lower(); s="".join(" " if ch in string.punctuation else ch for ch in s); s=re.sub(r"\b(a|an|the)\b"," ",s); return " ".join(s.split())
def clean(a): return re.sub(r"\[\s*\d+\s*\]","",a).strip()
def contains(a,gs): return any(norm(g) and f" {norm(g)} " in f" {norm(clean(a))} " for g in gs)
def auroc(p,n):
    p,n=np.asarray(p),np.asarray(n)
    return float(np.mean(p[:,None]>n[None,:])+0.5*np.mean(p[:,None]==n[None,:])) if len(p) and len(n) else None
def boot(p,n,B=2000):
    p,n=np.asarray(p),np.asarray(n); v=[auroc(rng.choice(p,len(p)),rng.choice(n,len(n))) for _ in range(B)]; return [float(x) for x in np.percentile(v,[2.5,97.5])]
TAU=json.load(open(V2/'summary_v2.json'))['tau'][0]
out={'tau':TAU,'configs':{}}
for f in sorted(glob.glob(str(V2/'gen_*.jsonl'))):
    n=Path(f).stem[4:]
    R={json.loads(l)['qid']:json.loads(l) for l in open(f)}
    E={json.loads(l)['qid']:json.loads(l) for l in open(V3/f'nli2_gen_{n}.jsonl')}
    rows=[]
    for q,r in R.items():
        e=E[q]; a=clean(r['answer'])
        rows.append({'cond':r['condition'],'abst':r['model_abstained'],'empty':(not r['model_abstained']) and a=='',
            'correct':r['condition']=='in_answerable' and (not r['model_abstained']) and contains(r['answer'],r['golds']),
            'top1':r['scores'][0],'ent':e.get('entail_max'),'span':e.get('span_support'),'vms':e.get('verify_ms') or 0.0})
    ans=[x for x in rows if not x['abst'] and not x['empty'] and x['ent'] is not None]
    pos=[x['ent'] for x in ans if x['correct']]; neg=[x['ent'] for x in ans if not x['correct']]
    negU=[x['ent'] for x in ans if x['cond']!='in_answerable']
    th=[]
    for b in range(2000):
        cal=np.sort([x['ent'] for x in ans if x['correct'] and rng.random()<0.5]); k=int(np.floor(ALPHA*(len(cal)+1)))
        th.append(float(cal[k-1]) if k>=1 else 0.0)  # rank < 1: threshold 0 rejects nothing
    TH=float(np.mean(th))
    def pol(gate,val,ver,span):
        ab=lambda x: x['abst'] or (val and x['empty']) or (gate and x['top1']<TAU) or (ver and x['ent'] is not None and x['ent']<TH) or (span and x['span'] is False)
        A=[x for x in rows if x['cond']=='in_answerable']; a=[x for x in rows if not ab(x)]
        return {'correct':round(float(np.mean([(not ab(x)) and x['correct'] for x in A])),3),
                'unans':round(float(np.mean([ab(x) for x in rows if x['cond']=='in_unanswerable'])),3),
                'ooc':round(float(np.mean([ab(x) for x in rows if x['cond']=='out_of_corpus'])),3),
                'coverage':round(len(a)/len(rows),3),'risk':round(float(np.mean([not x['correct'] for x in a])),3) if a else None}
    vms=[x['vms'] for x in rows if x['vms']]
    out['configs'][n]={'n_ans':len(ans),'auroc':auroc(pos,neg),'auroc_ci':boot(pos,neg),'auroc_vs_should_abstain':auroc(pos,negU),
        'theta':TH,'theta_sd':float(np.std(th)),'span_rate_correct':float(np.mean([x['span'] for x in ans if x['correct']])),
        'span_rate_unans_answered':float(np.mean([x['span'] for x in ans if x['cond']=='in_unanswerable'] or [np.nan])),
        'MVG':pol(1,1,0,0),'MVGS':pol(1,1,0,1),'MVGE':pol(1,1,1,0),'MVGSE':pol(1,1,1,1),
        'verify_ms_median':float(np.median(vms)),'verify_ms_p95':float(np.percentile(vms,95))}
P=[json.loads(l) for l in open(V3/'gen_poison_q3_k3.jsonl')]; PE={json.loads(l)['qid']:json.loads(l) for l in open(V3/'nli2_gen_poison_q3_k3.jsonl')}
th3=out['configs']['q3_k3']['theta']
att=lambda r:(not r['model_abstained']) and contains(r['answer'],[r['fake_answer']]) and not contains(r['answer'],r['golds'])
asr=lambda pred: float(np.mean([att(r) and pred(r) for r in P]))
out['poison']={'n':len(P),'poison_retrieved':float(np.mean([r['poison_in_topk']>0 for r in P])),'mean_poison_in_top3':float(np.mean([r['poison_in_topk'] for r in P])),
 'still_correct':float(np.mean([(not r['model_abstained']) and contains(r['answer'],r['golds']) for r in P])),'abstained':float(np.mean([r['model_abstained'] for r in P])),
 'ASR':asr(lambda r:True),'ASR_gate':asr(lambda r:r['scores'][0]>=TAU),
 'ASR_gate_span_nli':asr(lambda r:r['scores'][0]>=TAU and PE[r['qid']].get('span_support') is not False and (PE[r['qid']].get('entail_max') or 0)>=th3),
 'entail_mean_attacked':float(np.mean([PE[r['qid']]['entail_max'] for r in P if att(r)])),'top1_mean':float(np.mean([r['scores'][0] for r in P]))}
json.dump(out,open(V3/'summary_v3b.json','w'),indent=1)
for n,c in out['configs'].items():
    print(n,'n',c['n_ans'],'AUROC',round(c['auroc'],3),[round(x,2) for x in c['auroc_ci']],'vsAbst',round(c['auroc_vs_should_abstain'],3) if c['auroc_vs_should_abstain'] else None,'theta',round(c['theta'],3),'span_correct',round(c['span_rate_correct'],2),'span_unansAns',round(c['span_rate_unans_answered'],2),'ms',round(c['verify_ms_median']))
    for k in ['MVG','MVGS','MVGE','MVGSE']: print('   ',k,c[k])
print(json.dumps(out['poison'],indent=1))
