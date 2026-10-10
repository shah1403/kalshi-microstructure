import collections,io,json
from pathlib import Path
import zstandard

def records(path):
    with open(path,'rb') as b,zstandard.ZstdDecompressor().stream_reader(b) as z,io.TextIOWrapper(z) as f:
        for line in f:
            if line.strip():yield json.loads(line)

def audit(path):
    book=[{},{}]; last={}; prevtime=0; prevtrade=None
    c=collections.Counter(); examples={}; first=lastt=None; lags=[]; prices=[]
    for r in records(path):
        if isinstance(r,dict):continue
        kind=r[0];c['type_'+str(kind)]+=1
        if kind==5:
            c['status_'+r[3]]+=1
            continue
        t=r[4];first=t if first is None else first;lastt=t
        if t<prevtime:c['time_reversals']+=1
        prevtime=t
        if kind==0:book=[{},{}];last.pop("book",None)
        chain="book" if kind in (0,1) else "trade"
        if chain in last and r[2]!=last[chain]:
            c[chain+'_chain_mismatch']+=1;examples.setdefault('chain',(last[chain],r[:6]))
        last[chain]=r[3]
        if kind in (0,1):
            for s,p,q,*_ in r[6]:
                p=int(round(float(p)*10000));q=float(q)
                if q:book[s][p]=q
                else:book[s].pop(p,None)
        if kind==2:
            for side,p,q,tid,et in r[6]:
                c['trades']+=1;p=float(p)
                if book[0] and book[1]:
                    bid=max(book[0])/10000;ask=min(book[1])/10000
                    if bid>=ask:c['crossed_trade']+=1;continue
                    mid=(bid+ask)/2;prices.append(mid)
                    c['valid_trade']+=1
                    if .45<=mid<=.55:
                        c['balanced_trade']+=1
                        if ask-bid>.0188:c['balanced_spread_eligible']+=1
                    c['side_'+str(side)]+=1
                    if side==0 and p>=ask-1e-8 or side==1 and p<=bid+1e-8:c['aggressor_consistent']+=1
                    if side==0 and p<=bid+1e-8 or side==1 and p>=ask-1e-8:c['aggressor_opposite_quote']+=1
                    lags.append((t-et)/1e6)
    import numpy as np
    return dict(file=Path(path).name,first_receipt_ns=first,last_receipt_ns=lastt,hours=(lastt-first)/3.6e12,
                counts=dict(c),mid_range=[min(prices),max(prices)],lag_ms_quantiles=np.quantile(lags,[0,.5,.99,1]).tolist(),examples=examples)

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    out=[audit(p) for p in sorted((root/'inputs').glob('*.zst'))]
    (root/'input_audit.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
