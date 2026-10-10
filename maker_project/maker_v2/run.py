from pathlib import Path
import sys,json,gzip
from collections import Counter
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from maker_replay import Config,raw_events,parquet_events
from fees import Book,charge
from distributions import context,TRADE_H,NS
from data_and_scores import extract,flow_test
from quote_model import QuoteModel
from execution import DynamicQuote

ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent/'results'

def expected(p,bid,ask,c):
    margin=(ask-bid)/10000-charge(1,bid/10000,True,c)-charge(1,1-ask/10000,True,c)
    net=np.array(p['net_by_outcome']);net[3]=margin
    return float(np.dot(p['p'],net)),margin

def strategy(source,rows,flowpred,root,dataset,band,variant,sparse=False,c=None):
    c=c or Config();model=QuoteModel(root,dataset,rows);b=Book();i=-1;active=None;done=[];blocked=False
    total_cash=c.bankroll;decisions=[];counts=Counter();lastflow=None
    for e in source():
        t=e['t'];kind=e['kind']
        if kind=='gap':
            if e.get('block_book',True):blocked=True;b.valid=False
            if active:active.censored=True;active.stop(t,'data_gap');active.last_mark=None
            continue
        if kind=='book':
            if e['snapshot'] and b.t and (not sparse or t-b.t>5*NS):
                if active:active.censored=True;active.stop(t,'book_gap')
            b.update(e['levels'],t,e['snapshot'])
            if e['snapshot']:blocked=False
            if blocked:b.valid=False
            if not b.valid and active:active.censored=True;active.stop(t,'invalid_book')
        else:
            i+=1
            if sparse and t-b.t>5*NS:
                b.valid=False
                if active:active.censored=True;active.stop(t,'stale_book')
            if active and b.valid:active.trade(e,i)
        if active:
            active.step(t,i,b,kind=='book')
            if active.done:
                r=active.result();total_cash+=r['net'];done.append(r);active=None
        if kind!='trade' or not b.valid or blocked:continue
        if i in flowpred:lastflow=flowpred[i]
        if i%5:continue
        qt=b.quote()
        if not qt:continue
        bid,ask,mid,imb=qt;ctx=context(rows,i);model.update(t)
        fp=lastflow['forecast']['full'] if lastflow else None
        if active:
            if variant!='probability_dynamic' or active.stopping or fp is None:continue
            if not band[0]<=mid<=band[1]:active.stop(t,'odds_band_exit');continue
            direction=int(np.argmax(fp['mid'])-1)
            heldside=int(np.argmax(active.held)) if max(active.held)>1e-9 else None
            if heldside is not None:
                remaining=max(1,50-(i-active.n));hindex=max(0,np.searchsorted(TRADE_H,remaining,side='right')-1)
                need='buy' if heldside==0 else 'sell';adverse=fp['mid'][0 if heldside==0 else 2]
                if fp[need]['cdf_trades'][hindex]<.5 or adverse>.6:
                    active.stop(t,'opposite_flow_or_adverse_risk');continue
            p=model.predict(ctx,direction);ev,margin=expected(p,bid+100*direction,ask+100*direction,c)
            if heldside is None and (ev<=0 or margin<=.01):active.stop(t,'updated_negative_edge');continue
            desired=[bid+100*direction,ask+100*direction]
            if heldside is not None:
                basis=active.basis[heldside]/active.held[heldside]
                # Preserve at least 1c after two maker fees when completing inventory.
                pad=.01+charge(1,basis,True,c)+.0044
                if heldside==0:desired[1]=max(desired[1],int(np.ceil((basis+pad)*100))*100)
                else:desired[0]=min(desired[0],int(np.floor((1-basis-pad)*100))*100)
            for s,price in enumerate(desired):
                if not 0<price<10000:continue
                if (s==0 and price>=ask) or (s==1 and price<=bid):continue
                active.replace(s,price,t)
            continue
        if not band[0]<=mid<=band[1]:continue
        counts['band_decisions']+=1
        if total_cash<1.1:counts['cash_reject']+=1;continue
        if (ask-bid)/10000-charge(1,bid/10000,True,c)-charge(1,1-ask/10000,True,c)<=.01:continue
        counts['spread_decisions']+=1
        offset=0
        if variant!='spread_only_v2':
            if model.banks[0].n<30 or fp is None:counts['initializing']+=1;continue
            if variant=='probability_dynamic':offset=int(np.argmax(fp['mid'])-1)
            p=model.predict(ctx,offset);ev,margin=expected(p,bid+100*offset,ask+100*offset,c)
            decisions.append(dict(t=t,mid=mid,offset=offset,p=p,expected_net=ev))
            if p['n']<30 or ev<=0 or margin<=.01:counts['nonpositive_edge']+=1;continue
        active=DynamicQuote(t,i,bid+100*offset,ask+100*offset,c);counts['entries']+=1
    if active:
        active.censored=True;active.reason='end_of_recording';done.append(active.result())
    closed=[x for x in done if x['net'] is not None];unresolved=[x for x in done if x['net'] is None]
    curve=[];equity=c.bankroll;peak=equity;dd=0
    for r in done:
        for t,v in r['marks']:
            eq=equity+v-c.bankroll;curve.append([t,eq]);peak=max(peak,eq);dd=max(dd,peak-eq)
        if r['net'] is not None:equity+=r['net']
    out=dict(dataset=dataset,band=f'{band[0]:.2f}_{band[1]:.2f}',variant=variant,
         quotes=len(done),filled=sum(sum(r['filled'])>0 for r in done),paired=sum(r['matched']>=1-1e-9 for r in done),
         gross=sum(r['gross'] for r in closed),fees=sum(r['fees'] for r in closed),net=sum(r['net'] for r in closed),
         maxdd=dd,unresolved=len(unresolved),censored=sum(r['censored'] for r in done),
         replacement_requests=sum(a['action']=='replace_request' for r in done for a in r['actions']),
         counts=dict(counts))
    return out,dict(episodes=done,equity=curve,decisions=decisions)

def run():
    OUT.mkdir(exist_ok=True)
    datasets=[('KXMLBGAME_MLB',lambda:raw_events(next((ROOT/'inputs').glob('*MLB*.zst'))),False),
              ('KXNFLGAME_NFL',lambda:raw_events(next((ROOT/'inputs').glob('*NFL*.zst'))),False)]
    for s in ['BUF','JAC']:datasets.append(('BUFJAC_'+s,lambda s=s:parquet_events(ROOT/'inputs','KXNFLGAME-26JAN11BUFJAC-'+s),True))
    if len(sys.argv)>1:datasets=[d for d in datasets if d[0] in sys.argv[1:]]
    pnl=[]
    for name,source,sparse in datasets:
        print('EXTRACT',name,flush=True);rows,arb=extract(source,sparse)
        print('FLOW',name,len(rows),flush=True);scores,records,flowpred=flow_test(rows)
        # Arrival-order index avoids assigning a later same-timestamp forecast to an earlier trade.
        flowpred={r['i']:r for r in flowpred.values()}
        qm=QuoteModel(ROOT,name,rows);qs,qr=qm.accuracy()
        result=dict(dataset=name,sparse=sparse,arbitrage=arb,flow_accuracy=scores,quote_accuracy=qs)
        (OUT/(name+'_accuracy.json')).write_text(json.dumps(result))
        for label,recs in [('flow',records),('issued',list(flowpred.values())),('quote',qr)]:
            with gzip.open(OUT/(name+'_'+label+'_predictions.jsonl.gz'),'wt') as f:
                for r in recs:f.write(json.dumps(r)+'\n')
        print('SCORES',name,'full',scores['full'],'baseline',scores['all'],flush=True)
        for band in [(.45,.55),(.3,.7)]:
            for variant in ['spread_only_v2','probability_static','probability_dynamic']:
                print('PNL',name,band,variant,flush=True)
                s,ledger=strategy(source,rows,flowpred,ROOT,name,band,variant,sparse)
                pnl.append(s)
                (OUT/(name+'_'+s['band']+'_'+variant+'.json')).write_text(json.dumps(dict(summary=s,ledger=ledger)))
                print(json.dumps(s),flush=True)
    pnl=[json.loads(p.read_text())['summary'] for p in OUT.glob('*_0.*.json')]
    pd.DataFrame(pnl).to_csv(OUT/'pnl.csv',index=False)

if __name__=='__main__':run()
