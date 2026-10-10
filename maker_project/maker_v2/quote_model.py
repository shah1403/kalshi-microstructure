from pathlib import Path
import json,heapq,bisect
import numpy as np
from distributions import Bank,context,TRADE_H
from fees import charge
from maker_replay import Config

QWIDTH=114
QPRIOR=np.zeros(QWIDTH);QPRIOR[0]=1;QPRIOR[60]=1;QPRIOR[111]=1
# 0:5 outcome none / YES-only / NO-only / both-full / partial-pair
# 5:10 category-weighted net PnL; 10:61 first fill time; 61:112 opposite fill time
# 112,113 observed first/opposite seconds sums. Opposite targets risk-set only.
def vector(r):
    y=np.zeros(QWIDTH);yes,no=r['filled']
    cat=0 if yes+no<1e-9 else 3 if r['matched']>=1-1e-9 else 4 if min(yes,no)>0 else 1 if yes>0 else 2
    y[cat]=1;y[5+cat]=r['net']
    k=r['first_wait'];y[10+(k-1 if k is not None and 1<=k<=50 else 50)]=1
    if k is None:y[61:112]=np.nan;y[113]=np.nan
    else:
        j=r['opposite_wait'];y[61+(j-1 if j is not None and 1<=j<=50 else 50)]=1
        y[113]=r['opposite_wait_seconds'] or 0.
    if k is not None:
        first=next(f for f in r['fills'] if f['maker']);y[112]=(first['t']-r['t'])/1e9
    return y

class QuoteModel:
    def __init__(self,root,dataset,rows):
        self.banks={o:Bank(QWIDTH) for o in [-1,0,1]};self.pending=[];self.issued=[];self.times=[r['t'] for r in rows]
        self.rows=rows;unique={}
        for p in (root/'results').glob(dataset+'_*_episodes.jsonl'):
            for line in p.open():
                r=json.loads(line)
                if r['censored'] or r['mature_t'] is None or r['net'] is None:continue
                r['fees']=sum(charge(f['qty'],f['price'],f['maker'],Config()) for f in r['fills'])
                r['net']=r['gross']-r['fees']
                unique[(r['t'],r['offset'])]=r
        for r in unique.values():
            i=bisect.bisect_left(self.times,r['t'])
            if i>=len(rows) or not rows[i]['valid']:continue
            ctx=context(rows,i);self.pending.append((r['mature_t'],r['t'],r['offset'],ctx,vector(r),r))
        self.pending.sort(key=lambda z:(z[0],z[1],z[2]));self.ix=0
    def update(self,t):
        while self.ix<len(self.pending) and self.pending[self.ix][0]<t:
            _,_,offset,ctx,y,_=self.pending[self.ix];self.banks[offset].add(ctx,y);self.ix+=1
    def predict(self,ctx,offset=0):
        b=self.banks[offset];v=b.predict(ctx,prior=QPRIOR);p=v[:5]
        # Explicit outcome mixture. Both-filled margin can differ at a new quote,
        # so its component is replaced by that quote's planned net spread in policy.
        conditional_net=np.divide(v[5:10],p,out=np.zeros(5),where=p>1e-12)
        first=v[10:61];opp=v[61:112]
        return dict(n=b.n,p=p.tolist(),net_by_outcome=conditional_net.tolist(),
                    first_cdf=np.cumsum(first[:50])[TRADE_H-1].tolist(),opposite_cdf=np.cumsum(opp[:50])[TRADE_H-1].tolist(),
                    first_rmst=float(np.dot(np.r_[np.arange(1,51),50],first)),
                    opposite_rmst=float(np.dot(np.r_[np.arange(1,51),50],opp)),
                    first_mean_seconds_if_fill=float(v[112]/max(1-first[-1],1e-12)),
                    opposite_mean_seconds_if_fill=float(v[113]/max(1-opp[-1],1e-12)))
    def accuracy(self):
        """Fresh causal replay of held-out shadow quotes, not policy-selected fills."""
        bs={o:Bank(QWIDTH) for o in [-1,0,1]};ix=0;rec=[]
        for _,t,o,c,y,r in sorted(self.pending,key=lambda z:(z[1],z[2])):
            while ix<len(self.pending) and self.pending[ix][0]<t:
                _,_,off,c0,y0,_=self.pending[ix];bs[off].add(c0,y0);ix+=1
            if o or bs[o].n<30:continue
            forecasts={level:bs[o].predict(c,level,QPRIOR) for level in ['all','full']}
            actual=int(np.argmax(y[:5]));row=dict(t=t,mid=r['mid'],actual=actual,first_wait=r['first_wait'],opposite_wait=r['opposite_wait'])
            for level,v in forecasts.items():row[level]=dict(p=v[:5].tolist(),first_cdf=np.cumsum(v[10:60])[TRADE_H-1].tolist(),opposite_cdf=np.cumsum(v[61:111])[TRADE_H-1].tolist())
            rec.append(row)
        out={}
        for band,lo,hi in [('45_55',.45,.55),('30_70',.3,.7)]:
            rr=[r for r in rec if lo<=r['mid']<=hi];out[band]={}
            for level in ['all','full']:
                if not rr:out[band][level]=dict(n=0);continue
                p=np.array([r[level]['p'] for r in rr]);y=np.array([r['actual'] for r in rr]);hit=y!=0;both=y==3
                chosen=p.argmax(1);predfill=1-p[:,0];predboth=p[:,3]
                chosenfill=predfill>=.5
                f=np.array([r[level]['first_cdf'] for r in rr]);fy=np.array([[r['first_wait'] is not None and r['first_wait']<=h for h in TRADE_H] for r in rr])
                out[band][level]=dict(n=len(rr),outcome_accuracy=float(np.mean(chosen==y)),
                    any_fill_accuracy=float(np.mean(chosenfill==hit)),both_fill_accuracy=float(np.mean((predboth>=.5)==both)),
                    any_fill_brier=float(np.mean((predfill-hit)**2)),both_fill_brier=float(np.mean((predboth-both)**2)),
                    forecast_any_fill=float(predfill.mean()),actual_any_fill=float(hit.mean()),forecast_both=float(predboth.mean()),actual_both=float(both.mean()),
                    fill_precision=float(np.mean(hit[chosenfill])) if any(chosenfill) else None,
                    fill_recall=float(np.mean(chosenfill[hit])) if any(hit) else None,
                    first_by_h_brier=np.mean((f-fy)**2,axis=0).tolist(),
                    opposite_events=sum(r['opposite_wait'] is not None for r in rr))
        return out,rec
