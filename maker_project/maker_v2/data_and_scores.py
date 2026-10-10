from pathlib import Path
import sys,json,heapq,math
from collections import Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from maker_replay import Config,raw_events,parquet_events
from fees import Book,charge
from distributions import *

def pair_cost(book,q,c):
    """Depth-aware cost of BUY YES at asks and BUY NO at 1-YES-bids."""
    total=0.
    for side in (0,1):
        levels=book.levels[1-side];remaining=q
        for price,size in sorted(levels.items(),reverse=side==1):
            px=(price/10000 if side==0 else 1-price/10000)+c.taker_slippage
            take=min(remaining,size*c.depth_fraction)
            if take<=0:continue
            total+=take*px+charge(take,px,False,c);remaining-=take
            if remaining<1e-9:break
        if remaining>1e-9:return None
    return total

def extract(source,sparse=False):
    b=Book();rows=[];counts=Counter();blocked=False;seg=0;was_valid=False;midprev=.5;midstate=0
    episode=None;apparent=[];c=Config()
    for e in source():
        t=e['t'];counts[e['kind']]+=1
        if e['kind']=='gap':
            seg+=1
            if e.get('block_book',True):blocked=True;b.valid=False
            was_valid=False
            if episode:episode['end']=t;apparent.append(episode);episode=None
            continue
        if e['kind']=='book':
            if e['snapshot'] and b.t and (not sparse or t-b.t>5*NS):seg+=1
            b.update(e['levels'],t,e['snapshot'])
            if e['snapshot']:blocked=False
            if blocked:b.valid=False
            qt=b.quote() if b.valid else None
            if bool(qt)!=was_valid:seg+=1
            was_valid=bool(qt)
            if qt:
                if abs(qt[2]-midprev)>1e-9:midstate=int(np.sign(qt[2]-midprev))
                midprev=qt[2]
            # Audit crossed messages rather than silently treating them as profits.
            cost=pair_cost(b,1,c) if all(b.levels) else None
            if cost is not None and cost<1:
                counts['apparent_profitable_pair_updates']+=1
                if episode is None:episode=dict(start=t,min_cost=cost,valid=bool(qt),updates=0)
                episode['min_cost']=min(episode['min_cost'],cost);episode['updates']+=1
                if qt:counts['valid_executable_pair_updates']+=1
            elif episode:
                episode['end']=t;apparent.append(episode);episode=None
            continue
        if sparse and t-b.t>5*NS:
            if was_valid:seg+=1
            was_valid=False;b.valid=False
        qt=b.quote() if b.valid and not blocked else None
        bid,ask,mid,imb=(qt[0]/10000,qt[1]/10000,qt[2],qt[3]) if qt else (0.,1.,midprev,0.)
        rows.append(dict(t=t,side=e['side'],qty=e['qty'],price=e['price']/10000,seg=seg,valid=qt is not None,
                         mid=mid,imb=imb,midstate=midstate,bid=bid,ask=ask,
                         bq=b.quantity(0,qt[0]) if qt else 0.,aq=b.quantity(1,qt[1]) if qt else 0.))
    if episode:episode['end']=None;apparent.append(episode)
    return rows,dict(counts=counts,apparent_crossed_episodes=apparent,
                     certified_pairs=0,realized_arbitrage_pnl=0.,paired_settlement_value=0.,
                     note='Crossed reconstructed books are quarantined, not assumed executable.')

def metrics(records,level):
    if not records:return dict(n=0)
    p=np.array([r['forecast'][level]['side'] for r in records]);y=np.array([r['actual_side'] for r in records])
    s=np.array([r['forecast'][level]['size'] for r in records]);sy=np.array([r['actual_size'] for r in records])
    m=np.array([r['forecast'][level]['mid'] for r in records]);my=np.array([r['actual_mid'] for r in records])
    opp=np.array([r['forecast'][level]['opp']['cdf_trades'] for r in records])
    oy=np.array([[int(r['truth']['opp_trades'] is not None and r['truth']['opp_trades']<=h) for h in TRADE_H] for r in records])
    truth_wait=np.array([r['truth']['opp_trades'] or 50 for r in records]);pred_wait=np.array([r['forecast'][level]['opp']['restricted_mean_trades'] for r in records])
    counts=np.bincount(y,minlength=2);recall=[]
    for j in [0,1]:recall.append(float(np.mean(p.argmax(1)[y==j]==j)) if counts[j] else None)
    return dict(n=len(records),next_taker_accuracy=float(np.mean(p.argmax(1)==y)),
                buy_recall=recall[0],sell_recall=recall[1],balanced_accuracy=float(np.mean([x for x in recall if x is not None])),
                next_size_bin_accuracy=float(np.mean(s.argmax(1)==sy)),midpoint_state_accuracy=float(np.mean(m.argmax(1)==my)),
                midpoint_direction_accuracy=float(np.mean(m[:,[0,2]].argmax(1)[my!=1]==(my[my!=1]==2))) if any(my!=1) else None,
                midpoint_direction_n=int(sum(my!=1)),
                side_size_accuracy=float(np.mean(np.array([r['forecast'][level]['side_size'] for r in records]).argmax(1)==5*y+sy)),
                buy_volume_50_mae=float(np.mean([abs(r['forecast'][level]['buy_volume'][-1]-r['actual_buy_volume'][-1]) for r in records])),
                sell_volume_50_mae=float(np.mean([abs(r['forecast'][level]['sell_volume'][-1]-r['actual_sell_volume'][-1]) for r in records])),
                side_brier=float(np.mean((p[:,0]-(y==0))**2)),
                opposite_by_h_accuracy=np.mean((opp>=.5)==oy,axis=0).tolist(),opposite_by_h_brier=np.mean((opp-oy)**2,axis=0).tolist(),
                predicted_opposite_by_h=np.mean(opp,axis=0).tolist(),observed_opposite_by_h=np.mean(oy,axis=0).tolist(),
                capped_wait_mae=float(np.mean(abs(pred_wait-truth_wait))),
                predicted_capped_wait=float(pred_wait.mean()),observed_capped_wait=float(truth_wait.mean()),
                buy_fraction=float(np.mean(y==0)))

def block_ci(records,reps=1000):
    """Paired five-minute-block bootstrap for side accuracy improvement."""
    groups=defaultdict(lambda:[0.,0])
    for r in records:
        y=r['actual_side'];f=np.argmax(r['forecast']['full']['side'])==y;b=np.argmax(r['forecast']['all']['side'])==y
        key=r['t']//(300*NS);groups[key][0]+=int(f)-int(b);groups[key][1]+=1
    a=np.array(list(groups.values()))
    if len(a)<2:return None
    rng=np.random.default_rng(734);vals=[]
    for _ in range(reps):
        draw=a[rng.integers(0,len(a),len(a))].sum(axis=0);vals.append(draw[0]/draw[1])
    return dict(blocks=len(a),low=float(np.quantile(vals,.025)),high=float(np.quantile(vals,.975)))

def flow_test(rows):
    bank=Bank(WIDTH);pending=[];records=[];seq=0;predictions={};matured_times=[]
    for i,r in enumerate(rows):
        while pending and pending[0][0]<r['t']:
            _,_,ctx,lab=heapq.heappop(pending);bank.add(ctx,lab)
        if not r['valid']:continue
        # Feed model every fifth actual trade, independent of spread/odds/PnL.
        if i%5:continue
        ctx=context(rows,i);forecasts={}
        if bank.n>=100:
            for level in ['all','coarse','imb','full']:
                p=bank.predict(ctx,level,PRIOR)
                forecasts[level]=dict(side=p[SLICES['side']].tolist(),size=p[SLICES['size']].tolist(),
                    mid=p[SLICES['mid']].tolist(),side_size=p[SLICES['side_size']].tolist(),buy_size=p[SLICES['buy_size']].tolist(),sell_size=p[SLICES['sell_size']].tolist(),**{s:wait_stats(p,s) for s in ['buy','sell','opp']},
                    buy_volume=p[SLICES['buy_volume']].tolist(),sell_volume=p[SLICES['sell_volume']].tolist())
            predictions[i]=dict(i=i,t=r['t'],n=bank.n,ctx=ctx,forecast=forecasts)
        label=target(rows,i)
        if label is not None:
            if forecasts:
                y=label['y'];records.append(dict(i=i,t=r['t'],mid=r['mid'],last_side=r['side'],n=bank.n,
                    ctx=ctx,actual_side=int(y[SLICES['side']].argmax()),actual_size=int(y[SLICES['size']].argmax()),
                    actual_mid=int(y[SLICES['mid']].argmax()),actual_buy_volume=y[SLICES['buy_volume']].tolist(),actual_sell_volume=y[SLICES['sell_volume']].tolist(),truth=label['truth'],forecast=forecasts))
            heapq.heappush(pending,(label['mature'],seq,ctx,label['y']));seq+=1
    # Same issued predictions, scored in disjoint 50-trade windows as a check.
    disjoint=[];end=-1
    for r in records:
        if r['i']>end:disjoint.append(r);end=r['i']+50
    result={level:metrics(records,level) for level in ['all','coarse','imb','full']}
    result['disjoint']={level:metrics(disjoint,level) for level in ['all','full']}
    result['paired_accuracy_difference_95ci']=block_ci(records)
    result['persistence_accuracy']=float(np.mean([r['last_side']==r['actual_side'] for r in records])) if records else None
    result['scored_45_55']={level:metrics([r for r in records if .45<=r['mid']<=.55],level) for level in ['all','full']}
    result['scored_30_70']={level:metrics([r for r in records if .3<=r['mid']<=.7],level) for level in ['all','full']}
    return result,records,predictions
