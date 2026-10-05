import json, math, statistics, time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).parent
ROWS=json.loads((ROOT/'benchmark.json').read_text())
MODEL='sentence-transformers/all-MiniLM-L6-v2'

def toks(t):
    return {x for x in ''.join(c.lower() if c.isalnum() or c in '+#.' else ' ' for c in t).split() if len(x)>2}

def lexical(a,b):
    A,B=toks(a),toks(b)
    return len(A&B)/math.sqrt(len(A)*len(B)) if A and B else 0.0

def cls(labels,scores,thr):
    p=[int(s>=thr) for s in scores]
    tp=sum(x==1 and y==1 for x,y in zip(p,labels)); fp=sum(x==1 and y==0 for x,y in zip(p,labels))
    fn=sum(x==0 and y==1 for x,y in zip(p,labels)); tn=sum(x==0 and y==0 for x,y in zip(p,labels))
    precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    return {'threshold':round(thr,4),'precision':round(precision,4),'recall':round(recall,4),'f1':round(f1,4),'accuracy':round((tp+tn)/len(labels),4),'tp':tp,'fp':fp,'fn':fn,'tn':tn}

def choose(labels,scores):
    candidates=sorted(set([0.0,1.0]+[round(s,4) for s in scores]))
    return max((cls(labels,scores,t) for t in candidates),key=lambda m:(m['f1'],m['precision'],m['accuracy']))

def ranking(rows,scores):
    groups={}
    for r,s in zip(rows,scores): groups.setdefault(r['query'],[]).append((r,s))
    ranks=[]
    for q,items in groups.items():
        ranked=sorted(items,key=lambda x:x[1],reverse=True)
        rel=[i for i,(r,_) in enumerate(ranked,1) if r['label']==1]
        ranks.append(rel[0] if rel else None)
    n=len(ranks) or 1
    return {'recall_at_1':round(sum(r==1 for r in ranks)/n,4),'recall_at_3':round(sum(bool(r and r<=3) for r in ranks)/n,4),'mrr':round(sum((1/r if r else 0) for r in ranks)/n,4)}

dev=[r for r in ROWS if r['split']=='dev']; test=[r for r in ROWS if r['split']=='test']
lex_dev=[lexical(r['query'],r['passage']) for r in dev]; lex_test=[lexical(r['query'],r['passage']) for r in test]
model=SentenceTransformer(MODEL)

def sem(rows):
    times=[]; scores=[]
    for _ in range(3):
        t=time.perf_counter(); q=model.encode([r['query'] for r in rows],normalize_embeddings=True,show_progress_bar=False); p=model.encode([r['passage'] for r in rows],normalize_embeddings=True,show_progress_bar=False)
        scores=np.sum(q*p,axis=1).tolist(); times.append((time.perf_counter()-t)*1000/len(rows))
    return scores,statistics.median(times)

sem_dev,_=sem(dev); sem_test,lat=sem(test)
hy_dev=[.86*s+.14*l for s,l in zip(sem_dev,lex_dev)]; hy_test=[.86*s+.14*l for s,l in zip(sem_test,lex_test)]

def report(rows,scores,selected):
    preds=[int(s>=selected['threshold']) for s in scores]
    errors=[{'id':r['id'],'query':r['query'],'expected':r['label'],'predicted':p,'score':round(s,4)} for r,s,p in zip(rows,scores,preds) if p!=r['label']]
    return {'selected_on_dev':selected,'test':cls([r['label'] for r in rows],scores,selected['threshold']),'ranking':ranking(rows,scores),'errors':errors}

out={'model':MODEL,'dataset':{'total':len(ROWS),'dev':len(dev),'test':len(test)},'lexical':report(test,lex_test,choose([r['label'] for r in dev],lex_dev)),'semantic':report(test,sem_test,choose([r['label'] for r in dev],sem_dev)),'hybrid':report(test,hy_test,choose([r['label'] for r in dev],hy_dev)),'semantic_median_ms_per_pair':round(lat,3)}
(ROOT/'results.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))