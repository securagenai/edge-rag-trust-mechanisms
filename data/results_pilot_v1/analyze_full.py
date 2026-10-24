import json, numpy as np
exec(open('load.py').read())
rng=np.random.default_rng(0)
C=['in_answerable','in_unanswerable','out_of_corpus']
def ok_answer(r, mode='lenient'):
    return r['contains'] if mode=='lenient' else r['f1']>=0.5
def outcome(r, abstain, mode='lenient'):
    if r['condition']=='in_answerable':
        return (not abstain) and ok_answer(r,mode)
    return abstain
def auroc(pos,neg):
    pos,neg=np.array(pos),np.array(neg)
    return (np.mean(pos[:,None]>neg[None,:])+0.5*np.mean(pos[:,None]==neg[None,:]))
def boot_auc(pos,neg,B=2000):
    pos,neg=np.array(pos),np.array(neg); v=[]
    for _ in range(B):
        v.append(auroc(rng.choice(pos,len(pos)),rng.choice(neg,len(neg))))
    return np.percentile(v,[2.5,97.5])
S={c:[r['top1'] for r in rows if r['condition']==c] for c in C}
out={}
for neg in ['in_unanswerable','out_of_corpus']:
    a=auroc(S['in_answerable'],S[neg]); lo,hi=boot_auc(S['in_answerable'],S[neg])
    out[f'auroc_answerable_vs_{neg}']=(a,lo,hi)
a=auroc(S['in_answerable'],S['in_unanswerable']+S['out_of_corpus']); lo,hi=boot_auc(S['in_answerable'],S['in_unanswerable']+S['out_of_corpus'])
out['auroc_answerable_vs_both']=(a,lo,hi)
# model self-abstention as classifier (TPR/FPR)
for c in C:
    R=[r for r in rows if r['condition']==c]
    out[f'{c}_model_abstain']=np.mean([r['model_abstained'] for r in R])
    out[f'{c}_top1_mean_sd']=(np.mean(S[c]),np.std(S[c],ddof=1))
A=[r for r in rows if r['condition']=='in_answerable']
ans=[r for r in A if not r['model_abstained']]
out['answerable_answered_n']=len(ans)
out['answerable_correct_lenient_overall']=np.mean([outcome(r,r['model_abstained']) for r in A])
out['answerable_correct_strict_overall']=np.mean([outcome(r,r['model_abstained'],'strict') for r in A])
out['answerable_correct_lenient_given_answered']=np.mean([r['contains'] for r in ans])
out['gold_top1']=np.mean([r['gold_retrieved_top1'] for r in A])
out['gold_top3']=np.mean([r['gold_pid'] in r['retrieved_ids'] for r in A])
# correctness given gold retrieved
g=[r for r in ans if r['gold_pid'] in r['retrieved_ids']]
out['answered_correct_given_gold_in_top3']=(np.mean([r['contains'] for r in g]),len(g))
# conformal threshold: guarantee >= 1-alpha retention of answerable
def policy_eval(tau, R, use_model):
    res={}
    for c in C:
        Rc=[r for r in R if r['condition']==c]
        ab=[(r['top1']<tau) or (use_model and r['model_abstained']) for r in Rc]
        res[c]=np.mean(ab)  # abstention rate
    dec=[(r,(r['top1']<tau) or (use_model and r['model_abstained'])) for r in R]
    answered=[r for r,ab in dec if not ab]
    res['coverage']=len(answered)/len(R)
    res['risk']=np.mean([not outcome(r,False) for r in answered]) if answered else np.nan
    res['acc']=np.mean([outcome(r,ab) for r,ab in dec])
    return res
alpha=0.10
conf={k:[] for k in ['tau','ans_retained','unans_rejected','ooc_rejected','risk_gate','cov_gate','risk_comb','cov_comb','acc_gate','acc_comb']}
for b in range(2000):
    idx=rng.permutation(len(rows)); cal=[rows[i] for i in idx[:105]]; test=[rows[i] for i in idx[105:]]
    sc=np.sort([r['top1'] for r in cal if r['condition']=='in_answerable'])
    n=len(sc); k=int(np.floor(alpha*(n+1)))  # conformal lower quantile
    tau=sc[k-1] if k>=1 else -np.inf
    e=policy_eval(tau,test,False); e2=policy_eval(tau,test,True)
    conf['tau'].append(tau); conf['ans_retained'].append(1-e['in_answerable'])
    conf['unans_rejected'].append(e['in_unanswerable']); conf['ooc_rejected'].append(e['out_of_corpus'])
    conf['risk_gate'].append(e['risk']); conf['cov_gate'].append(e['coverage']); conf['acc_gate'].append(e['acc'])
    conf['risk_comb'].append(e2['risk']); conf['cov_comb'].append(e2['coverage']); conf['acc_comb'].append(e2['acc'])
out['conformal_alpha']=alpha
out['conformal']={k:(float(np.mean(v)),float(np.std(v))) for k,v in conf.items()}
# full-sample policies
out['policy_none']=policy_eval(-1,rows,False)
out['policy_model_only']=policy_eval(-1,rows,True)
tau_full=np.sort(S['in_answerable'])[int(np.floor(alpha*71))-1]
out['tau_full']=tau_full
out['policy_gate_full']=policy_eval(tau_full,rows,False)
out['policy_comb_full']=policy_eval(tau_full,rows,True)
# cost
def st(k): v=np.array([r[k] for r in rows if r.get(k) is not None]); return (v.mean(),np.median(v),np.percentile(v,95),v.max())
for k in ['retrieval_ms','generation_ms','prompt_tokens','completion_tokens','prompt_tok_per_s','gen_tok_per_s','server_rss_mb','process_rss_mb','mean_power_w']:
    out['cost_'+k]=st(k)
e=np.array([r['mean_power_w']*(r['retrieval_ms']+r['generation_ms'])/1000 for r in rows])
out['energy_J']=(e.mean(),np.median(e),np.percentile(e,95))
lat=np.array([(r['retrieval_ms']+r['generation_ms'])/1000 for r in rows]); out['latency_s']=(lat.mean(),np.median(lat),np.percentile(lat,95))
import pprint; pprint.pprint(out,width=120)
json.dump(out,open('analysis.json','w'),indent=1,default=float)
