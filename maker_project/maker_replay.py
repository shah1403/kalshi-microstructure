"""Counterfactual passive-order replay. Never connects to an account or submits orders.

Public market-by-price depth cannot identify cancel positions or our actual fills.
Primary queue model: all displayed volume ahead; only executions advance queue.
An alternative queue_fraction=0.5 is an optimistic sensitivity, NOT a fitted model.
"""
from collections import Counter,deque,defaultdict
from dataclasses import dataclass,asdict
from pathlib import Path
import json,math,heapq,hashlib
import numpy as np
from audit_raw import records

NS=1_000_000_000
EPS=1e-9

@dataclass
class Config:
    band_low:float=.45
    band_high:float=.55
    latency_ms:float=5.
    max_trades:int=50
    max_seconds:float=300.
    stride:int=5
    quantity:float=1.
    bankroll:float=5.
    min_net_margin:float=.01
    queue_fraction:float=1.
    maker_multiplier:float=1.
    taker_multiplier:float=1.
    fee_quantum:float=.0001
    taker_slippage:float=.001
    depth_fraction:float=.5
    warmup:int=30
    shrinkage:float=20.
    snapshot_max_age:float=5.

def charge(q,p,maker,c):
    rate=(.0175*c.maker_multiplier if maker else .07*c.taker_multiplier)
    return math.ceil(rate*q*p*(1-p)/c.fee_quantum-1e-10)*c.fee_quantum

class Book:
    def __init__(self):
        self.levels=[{},{}];self.valid=False;self.t=0
    def update(self,levels,t,snapshot=False):
        if snapshot:self.levels=[{},{}]
        for s,p,q in levels:
            if not (s in (0,1) and 0<p<10000 and math.isfinite(q) and q>=0):
                raise ValueError('invalid depth')
            if q:self.levels[s][p]=q
            else:self.levels[s].pop(p,None)
        self.t=t;self.valid=self.quote() is not None
    def quote(self):
        if not all(self.levels):return None
        b,a=max(self.levels[0]),min(self.levels[1])
        if b>=a:return None
        bq,aq=self.levels[0][b],self.levels[1][a]
        return b,a,(b+a)/20000,(bq-aq)/(bq+aq)
    def quantity(self,side,price):return self.levels[side].get(price,0.)
    def liquidate(self,side,q,c):
        # side 0: sell held YES at bids. side 1: sell held NO against YES asks.
        levels=self.levels[side]
        remaining=q;gross=fees=0.
        for p,v in sorted(levels.items(),reverse=side==0):
            take=min(remaining,v*c.depth_fraction)
            px=(p/10000 if side==0 else 1-p/10000)-c.taker_slippage
            if px<0:continue
            gross+=take*px;fees+=charge(take,px,False,c)
            remaining-=take
            if remaining<EPS:return gross,fees
        return None

class Episode:
    def __init__(self,idx,t,n,bid,ask,offset,key,coarse,mid,c):
        self.id=idx;self.t=t;self.start=n;self.c=c;self.offset=offset
        self.prices=[bid+offset*100,ask+offset*100]
        self.key=key;self.coarse=coarse;self.mid=mid
        self.arrival=t+round(c.latency_ms*1e6);self.active=False
        self.remaining=[c.quantity,c.quantity];self.ahead=[0.,0.]
        self.held=[0.,0.];self.cash=c.bankroll;self.gross=0.;self.fees=0.
        self.filled=[0.,0.];self.matched=0.;self.first_side=None
        self.first_n=None;self.first_t=None;self.second_n=None;self.second_t=None
        self.first_mid_sign=0;self.first_mid_delta=0.;self.adverse_t=None
        self.cancel_due=None;self.exit_due=None;self.expire_t=None
        self.flat_t=None;self.mature_t=None;self.censored=False;self.reason=None
        self.queue_at_arrival=None;self.fills=[];self.marks=[]
        self.peak=c.bankroll;self.maxdd=0.;self.last_mark=c.bankroll
        self.unknown_marks=0;self.terminal=False

    def cancel(self,t,reason):
        if self.cancel_due is None:
            self.cancel_due=t+round(self.c.latency_ms*1e6);self.reason=reason

    def maker_fill(self,side,q,t,n):
        if q<=EPS:return
        p=self.prices[side]/10000 if side==0 else 1-self.prices[side]/10000
        f=charge(q,p,True,self.c)
        self.cash-=q*p+f;self.fees+=f;self.held[side]+=q
        self.remaining[side]-=q;self.filled[side]+=q
        self.fills.append(dict(t=t,n=n,side=side,qty=q,price=p,maker=True,fee=f))
        if self.first_side is None:
            self.first_side=side;self.first_t=t;self.first_n=n
        elif side!=self.first_side and self.second_t is None:
            self.second_t=t;self.second_n=n
        paired=min(self.held)
        if paired>EPS:
            # Opposite purchase offsets same-market exposure; this is not final settlement.
            self.cash+=paired
            self.gross+=paired*(self.prices[1]-self.prices[0])/10000
            self.held=[h-paired for h in self.held];self.matched+=paired
        if max(self.remaining)<EPS:
            self.active=False;self.flat_t=t

    def trade(self,t,n,aggressor,p,qty,exchange_t):
        if not self.active or self.censored:return
        # A delayed public print from before order arrival cannot fill our order.
        if exchange_t and exchange_t<self.arrival:return
        if self.cancel_due is not None and t>=self.cancel_due:return
        side=1 if aggressor==0 else 0
        if self.remaining[side]<=EPS:return
        limit=self.prices[side]
        through=p>limit if side==1 else p<limit
        at=p==limit
        if not through and not at:return
        if through:
            self.ahead[side]=0.
            available=qty
        else:
            used=min(qty,self.ahead[side]);self.ahead[side]-=used;available=qty-used
        self.maker_fill(side,min(available,self.remaining[side]),t,n)

    def mark(self,t,book):
        if self.flat_t is not None and self.marks and t>self.flat_t:return
        if not book.valid:
            self.last_mark=None;self.unknown_marks+=1;return
        eq=self.cash
        for side,q in enumerate(self.held):
            if q>EPS:
                fill=book.liquidate(side,q,self.c)
                if fill is None:self.unknown_marks+=1;self.last_mark=None;return
                eq+=fill[0]-fill[1]
        self.last_mark=eq;self.peak=max(self.peak,eq);self.maxdd=max(self.maxdd,self.peak-eq)
        if not self.marks or abs(self.marks[-1][1]-eq)>1e-8:
            self.marks.append((t,eq))

    def step(self,t,n,book,is_book=False):
        if self.terminal:return
        qt=book.quote() if book.valid else None
        if qt and t>self.t and not self.first_mid_sign:
            delta=qt[2]-self.mid
            if abs(delta)>1e-9:self.first_mid_sign=int(np.sign(delta));self.first_mid_delta=delta
        if qt and self.first_side is not None and self.adverse_t is None:
            # At least one cent adverse to the first inventory, measured from its initial quote midpoint.
            delta=qt[2]-self.mid
            if (self.first_side==0 and delta<=-.01+EPS) or (self.first_side==1 and delta>=.01-EPS):
                self.adverse_t=t
        if self.expire_t is None and (n-self.start>=self.c.max_trades or t-self.t>=self.c.max_seconds*NS):
            self.expire_t=t;self.cancel(t,'50_trade' if n-self.start>=self.c.max_trades else '300_second')
        if self.censored:
            self.active=False
            if self.exit_due is None:self.exit_due=t+round(self.c.latency_ms*1e6)
        if not self.active and self.queue_at_arrival is None and not self.censored and t>=self.arrival:
            if qt and (self.prices[0]>=qt[1] or self.prices[1]<=qt[0]):
                self.reason='post_only_rejected';self.flat_t=t;self.queue_at_arrival=[]
            elif qt and self.cancel_due is None:
                self.ahead=[book.quantity(s,p)*self.c.queue_fraction for s,p in enumerate(self.prices)]
                self.queue_at_arrival=list(self.ahead);self.active=True
            elif not qt:
                self.censored=True;self.reason='arrival_book_missing'
        if self.cancel_due is not None and t>=self.cancel_due:
            self.active=False
            if self.exit_due is None:self.exit_due=self.cancel_due+round(self.c.latency_ms*1e6)
        if not self.active and max(self.held)<EPS and self.flat_t is None and (self.cancel_due is not None or self.censored):
            self.flat_t=t
        if self.exit_due is not None and is_book and book.valid and t>=self.exit_due and max(self.held)>EPS:
            for side,q in enumerate(self.held):
                if q<=EPS:continue
                fill=book.liquidate(side,q,self.c)
                if fill is None:continue
                proceeds,f=fill
                self.cash+=proceeds-f;self.fees+=f
                p=self.prices[side]/10000 if side==0 else 1-self.prices[side]/10000
                self.gross+=proceeds-q*p
                self.fills.append(dict(t=t,n=n,side=side,qty=q,price=proceeds/q,maker=False,fee=f))
                self.held[side]=0.
            if max(self.held)<EPS:self.flat_t=t
        self.mark(t,book)
        if self.expire_t is not None and self.flat_t is not None:
            self.mature_t=max(self.expire_t,self.flat_t);self.terminal=True

    def result(self):
        net=self.cash-self.c.bankroll if max(self.held)<EPS else None
        if net is not None:assert abs(net-(self.gross-self.fees))<1e-7
        return dict(id=self.id,t=self.t,start=self.start,offset=self.offset,key=self.key,coarse=self.coarse,
                    bid=self.prices[0]/10000,ask=self.prices[1]/10000,mid=self.mid,
                    first_side=self.first_side,first_wait=None if self.first_n is None else self.first_n-self.start,
                    opposite_wait=None if self.second_n is None else self.second_n-self.first_n,
                    opposite_wait_seconds=None if self.second_t is None else (self.second_t-self.first_t)/NS,
                    second_before_adverse=self.second_t is not None and (self.adverse_t is None or self.second_t<self.adverse_t),
                    filled=self.filled,matched=self.matched,held=self.held,queue_at_arrival=self.queue_at_arrival,
                    gross=self.gross,fees=self.fees,net=net,cash=self.cash,
                    mark_net=None if self.last_mark is None else self.last_mark-self.c.bankroll,
                    maxdd=self.maxdd,unknown_marks=self.unknown_marks,
                    censored=self.censored,reason=self.reason,flat_t=self.flat_t,mature_t=self.mature_t,
                    midpoint_state=self.first_mid_sign,midpoint_delta=self.first_mid_delta,
                    fills=self.fills,marks=self.marks)

def raw_events(path):
    prev={};seen=set();last_t=0
    for r in records(path):
        if isinstance(r,dict):continue
        if r[0]==5:
            if r[3]=='ERROR':yield dict(kind='gap',t=max(last_t,r[2]),reason=r[4])
            continue
        t=r[4];typ='book' if r[0] in (0,1) else 'trade'
        if t<last_t:
            yield dict(kind='gap',t=last_t,reason='receipt_time_reversal',block_book=typ=='book');continue
        last_t=t
        if r[0]==0:prev.pop('book',None)
        if typ in prev and r[2]!=prev[typ]:yield dict(kind='gap',t=t,reason=typ+'_chain_break',block_book=typ=='book')
        prev[typ]=r[3]
        if r[0] in (0,1):
            yield dict(kind='book',t=t,snapshot=r[0]==0,
                       levels=[(int(s),round(float(p)*10000),float(q)) for s,p,q,*_ in r[6]])
        else:
            for s,p,q,tid,et in r[6]:
                if tid in seen:continue
                seen.add(tid)
                yield dict(kind='trade',t=t,side=int(s),price=round(float(p)*10000),qty=float(q),exchange_t=et)

def parquet_events(root,ticker):
    import pandas as pd
    books=pd.read_parquet(root/'orderbook.parquet');trades=pd.read_parquet(root/'trades.parquet')
    events=[]
    for r in books[books.ticker==ticker].itertuples():
        levels=[]
        for side,field in [(0,r.yes_bids),(1,r.no_bids)]:
            for level in json.loads(field):
                px=level['price_cents']*100 if side==0 else (100-level['price_cents'])*100
                levels.append((side,px,float(level['size'])))
        events.append(dict(kind='book',t=r.ts.value,snapshot=True,levels=levels))
    for r in trades[trades.ticker==ticker].drop_duplicates('trade_id').itertuples():
        if r.side!='yes':raise ValueError('Unexpected parquet price convention')
        events.append(dict(kind='trade',t=r.ts.value,side=0 if r.taker_side=='yes' else 1,
                           price=r.price_cents*100,qty=float(r.count),exchange_t=r.ts.value))
    # Tied timestamps: trades before snapshots; don't use a future book to claim a fill.
    yield from sorted(events,key=lambda e:(e['t'],e['kind']=='book'))

def features(q,flow,prev_mid_state):
    b,a,mid,imb=q
    side=flow[-1][1] if flow else -1
    total=sum(x[2] for x in flow);signed=sum(x[2]*(1 if x[1]==0 else -1) for x in flow)
    f=signed/total if total else 0.
    rate=(len(flow)-1)*NS/max(1,flow[-1][0]-flow[0][0]) if len(flow)>1 else 0.
    coarse=(min(9,int(mid*10)),int(np.digitize(imb,[-.4,.4])),side,prev_mid_state)
    full=coarse+(int(np.digitize(f,[-.4,.4])),int(np.digitize(rate,[1.,10.])))
    return coarse,full,total,rate,f

def generate(events,c,sparse=False):
    book=Book();live=[];finished=[];anchors=[];flow=deque(maxlen=20)
    n=0;last_anchor=-100;aid=0;counts=Counter();last_t=0;prev_mid=None;mid_state=0
    blocked=False
    def gap(t,reason,block_book=True):
        nonlocal blocked
        blocked=blocked or block_book
        if block_book:book.valid=False
        counts['gap_'+reason]+=1
        for ep in live:
            ep.censored=True;ep.cancel(t,'data_gap');ep.reason='data_gap'
            if max(ep.held)>EPS:ep.last_mark=None;ep.unknown_marks+=1
        flow.clear()
    for ev in events:
        t=ev['t'];last_t=t;counts[ev['kind']]+=1
        if ev['kind']=='gap':gap(t,ev.get('reason','gap'),ev.get('block_book',True));continue
        if ev['kind']=='book':
            if not sparse and ev['snapshot'] and book.t:gap(t,'resnapshot')
            if sparse and book.t and t-book.t>c.snapshot_max_age*NS:gap(t,'sparse_snapshot_gap')
            book.update(ev['levels'],t,ev['snapshot'])
            if ev['snapshot']:blocked=False
            if blocked:book.valid=False
            qt=book.quote() if book.valid else None
            if qt:
                if prev_mid is not None and abs(qt[2]-prev_mid)>EPS:mid_state=int(np.sign(qt[2]-prev_mid))
                prev_mid=qt[2]
            else:
                counts['invalid_book']+=1
                for ep in live:ep.censored=True;ep.cancel(t,'invalid_book')
        else:
            n+=1
            if sparse and t-book.t>c.snapshot_max_age*NS and book.valid:gap(t,'stale_snapshot_on_trade')
            flow.append((t,ev['side'],ev['qty']))
            # Existing orders consume this print before new arrivals/decisions at this receipt time.
            if book.valid and not blocked:
                for ep in live:ep.trade(t,n,ev['side'],ev['price'],ev['qty'],ev['exchange_t'])
        for ep in live:ep.step(t,n,book,ev['kind']=='book')
        keep=[]
        for ep in live:
            if ep.terminal:finished.append(ep.result())
            else:keep.append(ep)
        live=keep
        if ev['kind']!='trade' or not book.valid or blocked:continue
        qt=book.quote()
        if not qt:continue
        b,a,mid,imb=qt
        counts['valid_trade']+=1
        if not c.band_low<=mid<=c.band_high:continue
        counts['balanced_trade']+=1
        margin=(a-b)/10000-charge(1,b/10000,True,c)-charge(1,1-a/10000,True,c)
        if margin<=c.min_net_margin+EPS:continue
        counts['spread_eligible_trade']+=1
        if n-last_anchor<c.stride:continue
        coarse,key,total,rate,f=features(qt,flow,mid_state)
        # Queue drain ratio is known from current depth and past trading volume only.
        depth=book.quantity(0,b)+book.quantity(1,a)
        key=key+(int(np.digitize(depth/max(total,1.),[1.,10.])),)
        anchors.append(dict(id=aid,t=t,n=n,mid=mid,imbalance=imb,spread=(a-b)/10000,
                            rate=rate,flow=f,depth=depth,key=key,coarse=coarse))
        for offset in (-1,0,1):
            p0,p1=b+offset*100,a+offset*100
            if not 0<p0<p1<10000:continue
            if (p1-p0)/10000-charge(1,p0/10000,True,c)-charge(1,1-p1/10000,True,c)<=c.min_net_margin+EPS:continue
            live.append(Episode(aid,t,n,b,a,offset,key,coarse,mid,c))
        aid+=1;last_anchor=n
    for ep in live:
        # End-of-recording is not a timeout or a fictitious close.
        ep.censored=True
        if ep.reason is None:ep.reason='end_of_recording'
        finished.append(ep.result())
    return anchors,finished,dict(counts)

class Rolling:
    def __init__(self,kappa=20.):
        self.kappa=kappa;self.data=defaultdict(lambda:np.zeros(12))
    def add(self,r):
        if r['censored'] or r['net'] is None or r['mature_t'] is None:return
        # [n, sum pnl, downfirst, nonefirst, upfirst, bothfull, anyfirst, opposite,
        #  sum opposite wait, midDown, midZero, midUp]
        vec=np.zeros(12);vec[0]=1;vec[1]=r['net']
        vec[3 if r['first_side'] is None else 4 if r['first_side']==0 else 2]=1
        vec[5]=r['matched']>=1-EPS;vec[6]=r['first_side'] is not None
        vec[7]=r['opposite_wait'] is not None;vec[8]=r['opposite_wait'] or 0
        vec[10+r['midpoint_state']]=1
        for k in [('all',r['offset']),('coarse',r['offset'],tuple(r['coarse'])),('full',r['offset'],tuple(r['key']))]:
            self.data[k]+=vec
    def predict(self,anchor,offset):
        a=self.data[('all',offset)];co=self.data[('coarse',offset,tuple(anchor['coarse']))];f=self.data[('full',offset,tuple(anchor['key']))]
        out=dict(n=int(a[0]),coarse_n=int(co[0]),cell_n=int(f[0]))
        ev=a[1]/(a[0]+self.kappa)
        p=(a[2:5]+.5)/(a[0]+1.5);mid=(a[9:12]+.5)/(a[0]+1.5)
        both=(a[5]+.5)/(a[0]+1)
        opposite=(a[7]+.5)/(a[6]+1)
        for v in (co,f):
            ev=(v[1]+self.kappa*ev)/(v[0]+self.kappa)
            p=(v[2:5]+self.kappa*p)/(v[0]+self.kappa)
            mid=(v[9:12]+self.kappa*mid)/(v[0]+self.kappa)
            both=(v[5]+self.kappa*both)/(v[0]+self.kappa)
            opposite=(v[7]+self.kappa*opposite)/(v[6]+self.kappa)
        return dict(**out,ev=float(ev),p=p.tolist(),mid=mid.tolist(),both=float(both),opposite=float(opposite))

def evaluate(anchors,episodes,c):
    lookup={(r['id'],r['offset']):r for r in episodes}
    learn=sorted([r for r in episodes if r['mature_t'] is not None and not r['censored']],key=lambda r:r['mature_t'])
    model=Rolling(c.shrinkage);ix=0;preds=[]
    names=['spread_only','conditioned_fill','conditioned_fill_skew','oracle_same_quotes','oracle_all_times']
    states={name:dict(cash=c.bankroll,busy=-1,selected=[],pending=None) for name in names}
    for a in anchors:
        while ix<len(learn) and learn[ix]['mature_t']<a['t']:
            model.add(learn[ix]);ix+=1
        p0=model.predict(a,0)
        pred={**a,**p0,'trade_n':a.get('n')}
        r0=lookup.get((a['id'],0))
        if r0:
            pred.update(realized_first=r0['first_side'],matched=r0['matched'],censored=r0['censored'],net=r0['net'])
        preds.append(pred)
        for name,st in states.items():
            if a['t']<=st['busy']:continue
            if st['pending'] is not None:
                st['cash']+=st['pending']['net'];st['pending']=None
            if st['cash']<1.1:continue
            offset=0
            if name not in ('spread_only','oracle_all_times') and p0['n']<c.warmup:continue
            if name=='conditioned_fill':
                if p0['ev']<=0:continue
            elif name=='conditioned_fill_skew':
                m=np.array(p0['mid']);w=np.flatnonzero(np.isclose(m,m.max()))
                offset=int(w[0]-1) if len(w)==1 else 0
                if model.predict(a,offset)['ev']<=0:continue
            elif name in ('oracle_same_quotes','oracle_all_times'):
                opts=[lookup.get((a['id'],o)) for o in (-1,0,1)]
                opts=[r for r in opts if r and r['net'] is not None and not r['censored']]
                if not opts:continue
                r=max(opts,key=lambda z:z['net'])
                if r['net']<=0:continue
                offset=r['offset']
            r=lookup.get((a['id'],offset))
            if not r:continue
            # Future outcome is read only for ledger bookkeeping, never to reject a causal strategy entry.
            st['selected'].append(r)
            st['busy']=r['flat_t'] if r['flat_t'] is not None else math.inf
            st['pending']=r
    summaries=[];ledgers={}
    for name,st in states.items():
        rows=st['selected'];equity=c.bankroll;peak=equity;dd=0.;curve=[];unknown=0
        for r in rows:
            for t,v in r['marks']:
                eq=equity+v-c.bankroll;peak=max(peak,eq);dd=max(dd,peak-eq);curve.append([t,eq])
            if r['net'] is not None:equity+=r['net']
            unknown+=r['unknown_marks']
        closed=[r for r in rows if r['net'] is not None]
        openr=[r for r in rows if r['net'] is None]
        net=sum(r['net'] for r in closed);fees=sum(r['fees'] for r in closed);gross=sum(r['gross'] for r in closed)
        assert abs(gross-fees-net)<1e-7
        waits=[r['opposite_wait'] for r in rows if r['opposite_wait'] is not None]
        fills=[r for r in rows if sum(r['filled'])>EPS]
        pnl_marks=net+sum((r['mark_net'] or 0) for r in openr)
        summaries.append(dict(strategy=name,quotes=len(rows),any_fill=len(fills),both_full=sum(r['matched']>=1-EPS for r in rows),
                              partial_pair=sum(EPS<r['matched']<1-EPS for r in rows),
                              taker_exits=sum(any(not f['maker'] for f in r['fills']) for r in rows),
                              gross=gross,fees=fees,net=net,net_with_last_marks=pnl_marks if all(r['mark_net'] is not None for r in openr) else None,
                              max_drawdown=dd,unresolved=len(openr),unknown_marks=unknown,
                              censored_selected=sum(r['censored'] for r in rows),
                              median_opposite_wait=float(np.median(waits)) if waits else None,
                              win_rate=sum(r['net']>EPS for r in fills if r['net'] is not None)/sum(r['net'] is not None for r in fills) if any(r['net'] is not None for r in fills) else None))
        ledgers[name]=dict(selected=rows,equity=curve)
    return summaries,preds,ledgers

def run(root):
    import pandas as pd
    datasets=[(p.stem.split('-')[1]+'_'+('MLB' if 'MLB' in p.name else 'NFL'),lambda p=p:raw_events(p),False) for p in sorted((root/'inputs').glob('*.zst'))]
    for suffix in ['JAC','BUF']:
        ticker='KXNFLGAME-26JAN11BUFJAC-'+suffix
        datasets.append(('BUFJAC_'+suffix,lambda ticker=ticker:parquet_events(root/'inputs',ticker),True))
    configs=[('primary_45_55',Config()),('diagnostic_30_70',Config(band_low=.3,band_high=.7))]
    out=root/'results';out.mkdir(exist_ok=True)
    all_summary=[];all_audit=[]
    for name,source,sparse in datasets:
        for label,c in configs:
            print('RUN',name,label,flush=True)
            anchors,episodes,audit=generate(source(),c,sparse)
            summaries,preds,ledgers=evaluate(anchors,episodes,c)
            prefix=out/(name+'_'+label)
            for r in summaries:r.update(dataset=name,scenario=label,sparse=sparse)
            all_summary.extend(summaries)
            diagnostics=dict(dataset=name,scenario=label,config=asdict(c),events=audit,anchors=len(anchors),
                             episodes=len(episodes),matured=sum(r['mature_t'] is not None and not r['censored'] for r in episodes),
                             censored=sum(r['censored'] for r in episodes),
                             episode_reasons=dict(Counter(r['reason'] for r in episodes)))
            all_audit.append(diagnostics)
            prefix.with_suffix('.json').write_text(json.dumps(dict(diagnostics=diagnostics,summary=summaries,predictions=preds,ledgers=ledgers),allow_nan=False))
            # All counterfactual labels are retained to audit the rolling data and oracle.
            with open(str(prefix)+'_episodes.jsonl','w') as f:
                for r in episodes:f.write(json.dumps(r,allow_nan=False)+'\n')
            print(json.dumps(summaries),flush=True)
    pd.DataFrame(all_summary).to_csv(out/'summary.csv',index=False)
    (out/'audit.json').write_text(json.dumps(all_audit,indent=2))

if __name__=='__main__':run(Path(__file__).resolve().parent)
