"""Expanding empirical marked-event distributions; no fitted regression."""
from collections import defaultdict,deque
import math
import numpy as np

TRADE_H=np.array([1,2,5,10,20,50])
SECOND_H=np.array([.005,.01,.05,.1,.25,.5,1.,2.,5.,10.,30.,60.,300.])
SIZE_EDGES=np.array([1.,10.,100.,1000.])
NS=10**9

def context(rows,i):
    r=rows[i];j=max(0,i-19)
    while j<i and rows[j]['seg']!=r['seg']:j+=1
    recent=rows[j:i+1];buy=[x for x in recent if x['side']==0];sell=[x for x in recent if x['side']==1]
    volume=[sum(x['qty'] for x in xs) for xs in (buy,sell)]
    sec=max((r['t']-recent[0]['t'])/NS,.001)
    countshare=len(buy)/len(recent);volshare=volume[0]/max(sum(volume),1e-9)
    rates=[len(buy)/sec,len(sell)/sec]
    since=[(r['t']-xs[-1]['t'])/NS if xs else 300. for xs in (buy,sell)]
    # Fixed bins: no sample-wide quantiles. Counts and volumes are separate.
    c=(min(9,int(r['mid']*10)),r['side'])
    im=c+(int(np.digitize(r['imb'],[-.4,.4])),r['midstate'])
    history=tuple(x['side'] for x in recent[-3:])
    f=im+(int(np.digitize(countshare,[.35,.65])),int(np.digitize(volshare,[.35,.65])),
          *[int(np.digitize(x,[1.,10.])) for x in rates],
          *[int(np.digitize(x,[.1,1.,10.])) for x in since],*history)
    queues=tuple(int(np.digitize(x,[1.,10.,100.])) for x in [r['bq']/max(volume[1],1.),r['aq']/max(volume[0],1.)])
    return dict(coarse=c,imb=im,full=f+queues+(int(np.digitize(r['ask']-r['bid'],[.015,.025,.05])),),
                buy_rate=rates[0],sell_rate=rates[1],count_buy_share=countshare,volume_buy_share=volshare,
                since_buy=since[0],since_sell=since[1])

class Bank:
    """Hierarchical empirical sums; denominator is tracked per target component."""
    def __init__(self,width,kappa=20):
        self.width=width;self.kappa=kappa;self.data=defaultdict(lambda:np.zeros((2,width)));self.n=0
    def keys(self,c):return [('all',),('coarse',tuple(c['coarse'])),('imb',tuple(c['imb'])),('full',tuple(c['full']))]
    def add(self,c,y):
        y=np.asarray(y);good=np.isfinite(y);v=np.where(good,y,0.)
        for k in self.keys(c):self.data[k][0]+=good;self.data[k][1]+=v
        self.n+=1
    def predict(self,c,level='full',prior=None):
        prior=np.zeros(self.width) if prior is None else np.asarray(prior)
        ks=self.keys(c);a=self.data[ks[0]]
        p=(a[1]+prior)/(a[0]+1.)
        if level!='all':
            end={'coarse':2,'imb':3,'full':4}[level]
            for k in ks[1:end]:
                v=self.data[k];p=(v[1]+self.kappa*p)/(v[0]+self.kappa)
        return p

# Fixed vector layout. Each timing outcome has 51 masses: trade 1..50, no arrival
# before the episode's 50-trade/300-second stop. Seconds CDF uses the same stop.
SLICES={};cursor=0
for name,size in [('side',2),('size',5),('side_size',10),('buy_size',5),('sell_size',5),('mid',3),('buy_n',51),('sell_n',51),('opp_n',51),
                  ('buy_s',13),('sell_s',13),('opp_s',13),('buy_sum_s',1),('sell_sum_s',1),('opp_sum_s',1),
                  ('buy_volume',6),('sell_volume',6)]:
    SLICES[name]=slice(cursor,cursor+size);cursor+=size
WIDTH=cursor
PRIOR=np.zeros(WIDTH)
for k in ['side','size','side_size','buy_size','sell_size','mid']:
    sl=SLICES[k];PRIOR[sl]=1/(sl.stop-sl.start)

for k in ['buy_n','sell_n','opp_n']:PRIOR[SLICES[k].stop-1]=1.

def target(rows,i):
    """Fully observed policy-window label, or None for data gaps/end of file."""
    r=rows[i];future=[];deadline=r['t']+300*NS;mature=None
    for k in range(i+1,min(len(rows),i+51)):
        x=rows[k]
        if x['seg']!=r['seg']:return None
        if x['t']>=deadline:mature=deadline;break
        future.append(x)
        if k==i+50:mature=x['t']
    if mature is None:return None
    if not future:return None
    y=np.zeros(WIDTH);y[SLICES['side']][future[0]['side']]=1
    size_bin=np.searchsorted(SIZE_EDGES,future[0]['qty'],side='right')
    y[SLICES['size']][size_bin]=1
    y[SLICES['side_size']][5*future[0]['side']+size_bin]=1
    change=next((int(np.sign(x['mid']-r['mid'])) for x in future if x['valid'] and abs(x['mid']-r['mid'])>1e-9),0)
    y[SLICES['mid']][change+1]=1
    truth={}
    for name,side in [('buy',0),('sell',1),('opp',1-r['side'])]:
        hit=next(((k+1,x) for k,x in enumerate(future) if x['side']==side),None)
        index=hit[0]-1 if hit else 50;y[SLICES[name+'_n']][index]=1
        dt=(hit[1]['t']-r['t'])/NS if hit else math.inf
        y[SLICES[name+'_s']]=np.array(dt<=SECOND_H,float)
        y[SLICES[name+'_sum_s']]=dt if hit else 0.
        truth[name+'_trades']=hit[0] if hit else None;truth[name+'_seconds']=dt if hit else None
        if name in ('buy','sell'):
            if hit:y[SLICES[name+'_size']][np.searchsorted(SIZE_EDGES,hit[1]['qty'],side='right')]=1
            else:y[SLICES[name+'_size']]=np.nan
    for name,side in [('buy',0),('sell',1)]:
        y[SLICES[name+'_volume']]=[sum(x['qty'] for x in future[:h] if x['side']==side) for h in TRADE_H]
    return dict(y=y,mature=mature,truth=truth)

def wait_stats(p,name):
    mass=p[SLICES[name+'_n']];cdf=np.cumsum(mass[:50]);hit=1-mass[-1]
    mean=float(np.dot(np.arange(1,51),mass[:50])/hit) if hit>1e-12 else None
    mean_s=float(p[SLICES[name+'_sum_s']][0]/hit) if hit>1e-12 else None
    # RMST includes no-arrival mass. This is not an unconditional eventual wait.
    rmst=float(np.dot(np.r_[np.arange(1,51),50],mass))
    return dict(probability_before_stop=float(hit),cdf_trades=cdf[TRADE_H-1].tolist(),
                cdf_seconds=p[SLICES[name+'_s']].tolist(),mean_trades_if_arrives=mean,
                mean_seconds_if_arrives=mean_s,restricted_mean_trades=rmst)
