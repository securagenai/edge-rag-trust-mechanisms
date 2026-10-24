import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('load.py').read())
plt.rcParams.update({'font.size':8,'font.family':'serif','axes.linewidth':0.6,'pdf.fonttype':42})
C=['in_answerable','in_unanswerable','out_of_corpus']
LAB={'in_answerable':'Answerable (in corpus)','in_unanswerable':'Unanswerable (in corpus)','out_of_corpus':'Out of corpus'}
COL={'in_answerable':'#1b6ca8','in_unanswerable':'#e08a1e','out_of_corpus':'#4f9a3c'}
S={c:np.array([r['top1'] for r in rows if r['condition']==c]) for c in C}
an=json.load(open('analysis.json')); tau=an['conformal']['tau'][0]
rng=np.random.default_rng(1)
# ---------- Fig 3 ----------
fig,ax=plt.subplots(1,2,figsize=(7.0,2.6))
for i,c in enumerate(C):
    y=S[c]; x=i+rng.uniform(-0.18,0.18,len(y))
    ax[0].scatter(x,y,s=7,alpha=0.6,color=COL[c],edgecolors='none')
    ax[0].boxplot(y,positions=[i],widths=0.5,showfliers=False,medianprops=dict(color='k',lw=1),
                  boxprops=dict(lw=0.7),whiskerprops=dict(lw=0.7),capprops=dict(lw=0.7))
ax[0].axhline(tau,ls='--',lw=0.8,color='k'); ax[0].text(2.45,tau+0.005,r'$\hat\tau$',ha='right',va='bottom')
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(['Answerable\n(in corpus)','Unanswerable\n(in corpus)','Out of\ncorpus'])
ax[0].set_ylabel('Top-1 retrieval score (cosine)'); ax[0].set_title('(a) Retrieval-score distributions',fontsize=8.5)
def roc(pos,neg):
    t=np.sort(np.unique(np.r_[pos,neg]))[::-1]
    tpr=[0]+[np.mean(pos>=x) for x in t]; fpr=[0]+[np.mean(neg>=x) for x in t]; return np.array(fpr),np.array(tpr)
for neg,key in [('out_of_corpus','auroc_answerable_vs_out_of_corpus'),('in_unanswerable','auroc_answerable_vs_in_unanswerable')]:
    f,t=roc(S['in_answerable'],S[neg]); a,lo,hi=an[key]
    ax[1].plot(f,t,color=COL[neg],lw=1.3,label=f'vs. {LAB[neg].lower()}: AUROC {a:.2f} [{lo:.2f}, {hi:.2f}]')
ax[1].plot([0,1],[0,1],':',color='gray',lw=0.8)
ax[1].set_xlabel('Should-abstain queries passing the gate'); ax[1].set_ylabel('Answerable queries passing the gate')
ax[1].set_title('(b) Retrieval-score gating as a classifier',fontsize=8.5); ax[1].legend(fontsize=6.5,loc='lower right',frameon=True,framealpha=0.9,edgecolor='none')
fig.tight_layout(); fig.savefig('Fig3.pdf'); fig.savefig('Fig3.png',dpi=200)
# ---------- Fig 4 ----------
def evalp(tau,use_model):
    ab={c:np.mean([(r['top1']<tau) or (use_model and r['model_abstained']) for r in rows if r['condition']==c]) for c in C}
    dec=[(r,(r['top1']<tau) or (use_model and r['model_abstained'])) for r in rows]
    ansd=[r for r,a in dec if not a]
    wrong=lambda r: not (r['condition']=='in_answerable' and r['contains'])
    risk=np.mean([wrong(r) for r in ansd]) if ansd else np.nan
    return ab,len(ansd)/len(rows),risk
pols=[('No abstention',-1,False),('Model self-abstention',-1,True),(r'Score gate ($\hat\tau$)',tau,False),('Gate + model',tau,True)]
fig,ax=plt.subplots(1,2,figsize=(7.0,2.6))
w=0.2
for j,(name,t,m) in enumerate(pols):
    ab,_,_=evalp(t,m)
    vals=[1-ab['in_answerable'],ab['in_unanswerable'],ab['out_of_corpus']]
    ax[0].bar(np.arange(3)+(j-1.5)*w,vals,w,label=name,color=['#bbbbbb','#7a7a7a','#1b6ca8','#0b2e4f'][j])
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(['Answerable\nretained','Unanswerable\nabstained','Out-of-corpus\nabstained'])
ax[0].set_ylim(0,1.32); ax[0].set_yticks([0,0.2,0.4,0.6,0.8,1.0]); ax[0].set_ylabel('Rate'); ax[0].legend(fontsize=6.3,frameon=False,ncol=2,loc='upper center')
ax[0].set_title('(a) Routing outcomes by query type',fontsize=8.5)
taus=np.linspace(0.33,0.76,200)
for m,ls,lab in [(False,'-','Score gate (sweep)'),(True,'--','Gate + model (sweep)')]:
    cov,rk=[],[]
    for t in taus:
        _,c,r=evalp(t,m)
        if c>0: cov.append(c); rk.append(r)
    ax[1].plot(cov,rk,ls,color='#1b6ca8' if not m else '#0b2e4f',lw=1.2,label=lab)
for name,t,m,mk in [('No abstention',-1,False,'o'),('Model self-abstention',-1,True,'x')]:
    _,c,r=evalp(t,m); ax[1].scatter([c],[r],marker=mk,color='k',s=22,label=name,zorder=3)
_,c,r=evalp(tau,True); ax[1].scatter([c],[r],marker='*',color='#d62728',s=40,zorder=4,label=r'Gate + model at $\hat\tau$')
ax[1].set_xlabel('Coverage (fraction of queries answered)'); ax[1].set_ylabel('Risk among answered queries')
ax[1].set_title('(b) Risk–coverage trade-off',fontsize=8.5); ax[1].legend(fontsize=6.3,frameon=False,loc='lower right')
ax[1].set_xlim(0,1.02); ax[1].set_ylim(0.3,1.0)
fig.tight_layout(); fig.savefig('Fig4.pdf'); fig.savefig('Fig4.png',dpi=200)
print('tau',tau)
