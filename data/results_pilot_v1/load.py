import json,re,string,collections
rows=[json.loads(l) for l in open('../results/results.jsonl')]
def norm(s):
    s=s.lower(); s=''.join(ch if ch not in string.punctuation else ' ' for ch in s)
    s=re.sub(r'\b(a|an|the)\b',' ',s); return ' '.join(s.split())
def f1(p,g):
    p,g=norm(p).split(),norm(g).split()
    c=sum((collections.Counter(p)&collections.Counter(g)).values())
    if c==0: return 0.
    pr,rc=c/len(p),c/len(g); return 2*pr*rc/(pr+rc)
for r in rows:
    a=re.sub(r'\[\s*\d+\s*\]','',r['answer']).strip()
    r['clean']=a; r['ntok']=len(norm(a).split())
    r['f1']=max([f1(a,g) for g in r['golds']] or [0])
    r['contains']=any(norm(g) and (' '+norm(g)+' ') in (' '+norm(a)+' ') for g in r['golds'])
    r['top1']=r['scores'][0]
