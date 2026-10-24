import json,re,string,glob,numpy as np
from pathlib import Path
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (HERE if (HERE / "results_v2").exists() else HERE.parent / "data")
os.chdir(DATA)
rng=np.random.default_rng(1)
def norm(s):
    s=s.lower(); s="".join(" " if ch in string.punctuation else ch for ch in s); s=re.sub(r"\b(a|an|the)\b"," ",s); return " ".join(s.split())
def clean(a): return re.sub(r"\[\s*\d+\s*\]","",a).strip()
def cont(a,gs): return any(norm(g) and f" {norm(g)} " in f" {norm(clean(a))} " for g in gs)
def f1(p,g):
    import collections
    p,g=norm(p).split(),norm(g).split(); c=sum((collections.Counter(p)&collections.Counter(g)).values())
    if c==0: return 0
    pr,rc=c/len(p),c/len(g); return 2*pr*rc/(pr+rc)
TAU=json.load(open('results_v2/summary_v2.json'))['tau'][0]
Q={q['qid']:q for q in json.load(open('questions_v2.json'))}
def load(n):
    R=[json.loads(l) for l in open(f'results_v2/gen_{n}.jsonl')]
    E={json.loads(l)['qid']:json.loads(l) for l in open(f'results_v3/nli2_gen_{n}.jsonl')}
    D={}
    for r in R:
        a=clean(r['answer']); e=E[r['qid']]
        D[r['qid']]=dict(cond=r['condition'],abst=r['model_abstained'],empty=(not r['model_abstained']) and a=='',
          len_=cont(r['answer'],r['golds']) and r['condition']=='in_answerable' and not r['model_abstained'],
          strict=(r['condition']=='in_answerable' and not r['model_abstained'] and max([f1(a,g) for g in r['golds']] or [0])>=0.5),
          top1=r['scores'][0],ent=e.get('entail_max'),hyp=e.get('hypothesis'))
    return D
C={n:load(n) for n in ['q15_k3','q3_k1','q3_k3','q3_k5','q7_k3']}
qids=list(Q)
def ab(x,V,G,E,th=0.0):
    return x['abst'] or (V and x['empty']) or (G and x['top1']<TAU) or (E and x['ent'] is not None and x['ent']<th)
def metrics(D,ids,V,G,E=0,th=0.0):
    A=[i for i in ids if D[i]['cond']=='in_answerable']; U=[i for i in ids if D[i]['cond']=='in_unanswerable']
    ans=[i for i in ids if not ab(D[i],V,G,E,th)]
    return dict(correct=np.mean([(not ab(D[i],V,G,E,th)) and D[i]['len_'] for i in A]),
                unans=np.mean([ab(D[i],V,G,E,th) for i in U]),
                risk=np.mean([not D[i]['len_'] for i in ans]) if ans else np.nan)
def paired(f,B=4000):
    pt=f(qids); bs=[]
    for _ in range(B):
        s=list(rng.choice(qids,len(qids))); bs.append(f(s))
    lo,hi=np.nanpercentile(bs,[2.5,97.5]); return round(float(pt),3),round(float(lo),3),round(float(hi),3)
out={}
# 3B vs 7B
out['7B-3B lenient']=paired(lambda s: np.mean([C['q7_k3'][i]['len_'] for i in s if Q[i]['condition']=='in_answerable'])-np.mean([C['q3_k3'][i]['len_'] for i in s if Q[i]['condition']=='in_answerable']))
out['7B-3B strict']=paired(lambda s: np.mean([C['q7_k3'][i]['strict'] for i in s if Q[i]['condition']=='in_answerable'])-np.mean([C['q3_k3'][i]['strict'] for i in s if Q[i]['condition']=='in_answerable']))
out['7B-3B risk M+V+G']=paired(lambda s: metrics(C['q7_k3'],s,1,1)['risk']-metrics(C['q3_k3'],s,1,1)['risk'])
out['k5-k3 lenient']=paired(lambda s: np.mean([C['q3_k5'][i]['len_'] for i in s if Q[i]['condition']=='in_answerable'])-np.mean([C['q3_k3'][i]['len_'] for i in s if Q[i]['condition']=='in_answerable']))
# discordant pairs 3B vs 7B lenient
A=[i for i in qids if Q[i]['condition']=='in_answerable']
out['3B_only_correct']=sum(C['q3_k3'][i]['len_'] and not C['q7_k3'][i]['len_'] for i in A); out['7B_only_correct']=sum(C['q7_k3'][i]['len_'] and not C['q3_k3'][i]['len_'] for i in A)
for n in ['q15_k3','q7_k3']:
    out[f'{n} +V risk change']=paired(lambda s,n=n: metrics(C[n],s,1,0)['risk']-metrics(C[n],s,0,0)['risk'])
for n in ['q3_k3','q7_k3']:
    out[f'{n} +G risk change']=paired(lambda s,n=n: metrics(C[n],s,1,1)['risk']-metrics(C[n],s,1,0)['risk'])
    out[f'{n} +G correct change']=paired(lambda s,n=n: metrics(C[n],s,1,1)['correct']-metrics(C[n],s,1,0)['correct'])
    out[f'{n} +G unans change']=paired(lambda s,n=n: metrics(C[n],s,1,1)['unans']-metrics(C[n],s,1,0)['unans'])
th=json.load(open('results_v3/summary_v3b.json'))['configs']
for n in ['q3_k3','q7_k3']:
    t=th[n]['theta']
    out[f'{n} +E unans change']=paired(lambda s,n=n,t=t: metrics(C[n],s,1,1,1,t)['unans']-metrics(C[n],s,1,1)['unans'])
    out[f'{n} +E correct change']=paired(lambda s,n=n,t=t: metrics(C[n],s,1,1,1,t)['correct']-metrics(C[n],s,1,1)['correct'])
# validator breakdown counts (all 300)
for n in ['q15_k3','q3_k3','q7_k3']:
    D=C[n]; out[f'{n} breakdown']=dict(substantive_correct=sum(D[i]['len_'] and not D[i]['empty'] for i in qids),
        substantive_incorrect=sum((not D[i]['abst']) and (not D[i]['empty']) and not D[i]['len_'] for i in qids),
        explicit_abstain=sum(D[i]['abst'] for i in qids), empty=sum(D[i]['empty'] for i in qids))
# per-split held-out gate
S=[(i,C['q3_k3'][i]['top1'],Q[i]['condition']) for i in qids]
ret=[];ooc=[];un=[];taus=[]
for b in range(2000):
    perm=rng.permutation(len(S)); cal=[S[j] for j in perm[:150]]; te=[S[j] for j in perm[150:]]
    sc=np.sort([x[1] for x in cal if x[2]=='in_answerable']); k=int(np.floor(0.1*(len(sc)+1))); t=sc[k-1] if k>=1 else -1.0; taus.append(t)
    ret.append(np.mean([x[1]>=t for x in te if x[2]=='in_answerable'])); ooc.append(np.mean([x[1]<t for x in te if x[2]=='out_of_corpus'])); un.append(np.mean([x[1]<t for x in te if x[2]=='in_unanswerable']))
out['split_gate']=dict(tau_mean=float(np.mean(taus)),tau_sd=float(np.std(taus)),ret_mean=float(np.mean(ret)),ret_p2_5=float(np.percentile(ret,2.5)),frac_ret_ge_90=float(np.mean(np.array(ret)>=0.9)),ooc_mean=float(np.mean(ooc)),unans_mean=float(np.mean(un)))
json.dump(out,open("results_v3/paired_stats.json","w"),indent=1,default=float); print("saved")
