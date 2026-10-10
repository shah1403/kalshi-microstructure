"""Causal replay of a live capture, NOT exchange execution or a live-order bot.

One shared $100 wallet and at most one funded strategy/market at a time.
Shadow quotes are forecasts, never booked as portfolio profit. No networking.
"""
import argparse, gzip, json, math, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'kalshi_paper'), str(ROOT/'maker_project/maker_v2'), str(ROOT/'maker_project')]
from paper_engine import Engine, Settings, Feed, Book as RawBook, fee, value
from maker_replay import Config
from fees import Book, charge
from execution import DynamicQuote
from distributions import Bank, context, target, WIDTH, PRIOR, SLICES, wait_stats
from data_and_scores import pair_cost

NS = 10**9

def one_tick_quotes(bid, ask, tick):
    """Center one-tick spread on the visible midpoint; don't claim new priority at existing prices."""
    if ask-bid < tick:
        return None
    qb = ((bid+ask-tick)//(2*tick))*tick
    qa = qb+tick
    if not (0 < qb < ask and bid < qa < 10000):
        return None
    return qb, qa

class Wallet:
    def __init__(self):
        self.cash = 100.
        self.owner = None
        self.peak = 100.
        self.dd = 0.
        self.curve = []
        self.unknown_marks = 0

class SharedTaker(Engine):
    def __init__(self, cfg, mult, wallet):
        super().__init__(cfg, mult)
        self.wallet = wallet
        self.tag = ('taker', cfg.ticker)
        self.drain = False

    def anchor(self, t, quote):
        self.cash = self.wallet.cash
        super().anchor(t, quote)
        if self.order is not None:
            if self.wallet.owner not in (None, self.tag) or self.drain:
                self.order = None
                self.diagnostics['shared_wallet_or_stop_block'] += 1
            else:
                self.wallet.owner = self.tag

    def execution(self, t, is_book):
        self.cash = self.wallet.cash
        before = self.cash
        super().execution(t, is_book)
        self.wallet.cash += self.cash-before
        if self.wallet.owner == self.tag and self.order is None and self.position is None:
            self.wallet.owner = None

class Market:
    def __init__(self, ticker, meta, wallet):
        self.ticker = ticker
        market, series = meta['market'], meta['series']
        mult = float(series['fee_multiplier'])
        self.c = Config(maker_multiplier=mult if series['fee_type']=='quadratic_with_maker_fees' else 0,
                        taker_multiplier=mult)
        ranges = market.get('price_ranges', [])
        self.tick = round(float(ranges[0]['step'])*10000) if len(ranges)==1 else None
        self.e = SharedTaker(Settings(ticker=ticker), mult, wallet)
        self.feed = Feed(self.e)
        self.feed.event_ticker = market['event_ticker']
        self.e.connected = False
        self.b = Book()
        self.raw_book = RawBook()
        self.rows = []
        self.seg = 0
        self.last_mid = None
        self.midstate = 0
        self.shadows = []
        self.shadow_done = []
        self.quote_bank = Bank(5)
        self.flow_bank = Bank(WIDTH)
        self.flow_pending = []
        self.flow_scores = []
        self.flow_prediction = None
        self.active = None
        self.ledger = []
        self.decisions = []
        self.audit = Counter()
        self.series = series
        self.observed_band_trades = 0

    def sync_book(self, t):
        self.b.levels = [dict(self.e.book.levels['yes']),
                         {10000-p:q for p,q in self.e.book.levels['no'].items()}]
        self.b.t = round(self.e.book.last_t*NS) if math.isfinite(self.e.book.last_t) else 0
        self.b.valid = bool(self.e.connected and self.e.tradable and self.b.quote()
                            and t-self.e.book.last_t <= self.e.cfg.max_book_age_s)

    def invalidate(self, t, reason):
        self.seg += 1
        self.e.gap(t, reason)
        self.b.valid = False
        self.flow_pending.clear()
        self.flow_prediction = None
        for sh in self.shadows:
            sh['quote'].censored = True
            sh['quote'].stop(round(t*NS), reason)
        if self.active:
            self.active.censored = True
            self.active.stop(round(t*NS), reason)
        self.audit[reason] += 1

    def update_flow(self, t):
        keep = []
        for item in self.flow_pending:
            lab = target(self.rows, item['i'])
            if lab is None or lab['mature'] >= t:
                keep.append(item)
                continue
            self.flow_bank.add(item['ctx'], lab['y'])
            if item['p']:
                actual_side = int(lab['y'][SLICES['side']].argmax())
                opp5 = lab['truth']['opp_trades'] is not None and lab['truth']['opp_trades'] <= 5
                self.flow_scores.append(dict(t=item['t'], actual_side=actual_side, opp5=opp5,
                    repeat=item['last_side'], predictions=item['p']))
        self.flow_pending = keep

    def on_trade(self, event):
        qt = self.b.quote() if self.b.valid else None
        bid, ask, mid, imb = qt if qt else (0,10000,self.last_mid or .5,0)
        if qt and self.last_mid is not None and mid != self.last_mid:
            self.midstate = int(np.sign(mid-self.last_mid))
        if qt:
            self.last_mid = mid
            if .45 <= mid <= .55:self.observed_band_trades += 1
        self.rows.append(dict(t=event['t'], side=event['side'], qty=event['qty'], price=event['price']/10000,
            seg=self.seg, valid=bool(qt), bid=bid/10000, ask=ask/10000, mid=mid, imb=imb,
            midstate=self.midstate, bq=self.b.quantity(0,bid), aq=self.b.quantity(1,ask)))
        self.update_flow(event['t'])
        i = len(self.rows)-1
        if not qt or i%5:
            return None
        ctx = context(self.rows, i)
        preds = {}
        if self.flow_bank.n >= 100:
            for level in ['all','full']:
                p = self.flow_bank.predict(ctx, level, PRIOR)
                preds[level] = dict(side=p[SLICES['side']].tolist(), mid=p[SLICES['mid']].tolist(),
                    opp5=wait_stats(p,'opp')['cdf_trades'][2],
                    buy50=wait_stats(p,'buy')['cdf_trades'][-1],
                    sell50=wait_stats(p,'sell')['cdf_trades'][-1])
            self.flow_prediction = preds['full']
        self.flow_pending.append(dict(i=i,t=event['t'],ctx=ctx,p=preds,last_side=event['side']))
        return ctx

    def process_quotes(self, t, trade, is_book, wallet, draining):
        n = self.e.count
        for sh in self.shadows:
            q = sh['quote']
            q.step(t,n,self.b,is_book)
            if trade and self.b.valid and not q.censored:
                q.trade(trade,n)
            q.step(t,n,self.b,is_book)
        keep = []
        for sh in self.shadows:
            q = sh['quote']
            # Observe the entire quote window, even if two fills happened early.
            matured = n-q.n >= 50 or t-q.t >= 300*NS
            if q.done and matured and q.closed_t < t:
                r=q.result()
                if not q.censored:
                    pair=float(q.matched >= 1-1e-9)
                    anyfill=float(sum(q.filled)>1e-9)
                    one=float(anyfill and not pair)
                    self.quote_bank.add(sh['ctx'], [pair,r['net'],anyfill,max(0,-r['net'])*one,one])
                    self.shadow_done.append(dict(t=q.t,net=r['net'],paired=bool(pair),filled=bool(anyfill),
                        predicted=sh.get('predicted')))
            else:
                keep.append(sh)
        self.shadows=keep
        if self.active:
            q=self.active
            if draining:
                q.stop(t,'pilot_stop')
            qt=self.b.quote() if self.b.valid else None
            if qt and not .45<=qt[2]<=.55:
                q.stop(t,'odds_band_exit')
            fp=self.flow_prediction
            if fp and max(q.held)>1e-9:
                held=int(np.argmax(q.held))
                if fp['buy50' if held==0 else 'sell50']<.5 or fp['mid'][0 if held==0 else 2]>.6:
                    q.stop(t,'adverse_flow')
            q.step(t,n,self.b,is_book)
            if trade and self.b.valid and not q.censored:
                q.trade(trade,n)
            q.step(t,n,self.b,is_book)
            if q.done:
                wallet.cash += q.cash
                wallet.owner = None
                self.ledger.append(q.result())
                self.active = None

    def decide_maker(self, t, ctx, wallet, draining):
        qt=self.b.quote() if self.b.valid else None
        if not qt or not .45<=qt[2]<=.55 or self.tick is None:
            self.audit['maker_outside_band_or_unsupported_grid']+=1
            return
        self.audit['maker_band_decisions']+=1
        prices=one_tick_quotes(qt[0],qt[1],self.tick)
        if not prices:
            return
        bid,ask=prices
        self.audit['compression_opportunities' if ask-bid < qt[1]-qt[0] else 'existing_one_tick_spread']+=1
        margin=(ask-bid)/10000-charge(1,bid/10000,True,self.c)-charge(1,1-ask/10000,True,self.c)
        pred=self.quote_bank.predict(ctx,prior=[0,0,0,0,0])
        loss=pred[3]/pred[4] if pred[4]>1e-12 else None
        ready=self.quote_bank.n>=30
        gate=ready and pred[1]>0 and margin>0 and (pred[0]>=.98 or (loss is not None and loss<=.005))
        d=dict(t=t,n=self.quote_bank.n,prices=[bid/10000,ask/10000],net_pair_margin=margin,
               probability_both=float(pred[0]),expected_net=float(pred[1]),expected_one_sided_loss=loss,
               ready=ready,qualifies=bool(gate),taken=False)
        if not draining:
            shadow=DynamicQuote(t,self.e.count,bid,ask,self.c)
            self.shadows.append(dict(quote=shadow,ctx=ctx,predicted=d.copy() if ready else None))
        if gate and wallet.owner is None and wallet.cash>=self.c.bankroll and not draining:
            self.active=DynamicQuote(t,self.e.count,bid,ask,self.c)
            wallet.cash-=self.c.bankroll
            wallet.owner=('maker',self.ticker)
            d['taken']=True
        self.decisions.append(d)

def run(directory, allow_incomplete=False):
    directory=Path(directory)
    summary=json.loads((directory/'capture_summary.json').read_text())
    wallet=Wallet()
    markets={k:Market(k,v,wallet) for k,v in summary['metadata'].items()}
    incomplete=bool(summary.get('incomplete',False))
    if incomplete and not allow_incomplete:
        raise RuntimeError('Interrupted capture requires explicit --allow-incomplete acknowledgment')
    end_metadata=summary.get('end_metadata')
    if not end_metadata and not (incomplete and allow_incomplete):
        raise RuntimeError('Missing closing fee metadata')
    if end_metadata:
        for k,m in markets.items():
            end=end_metadata[k]['series']
            if any(end.get(f)!=m.series.get(f) for f in ('fee_type','fee_multiplier')):
                raise RuntimeError('Fee metadata changed; discard pilot and record again')
    # The recorder duration is authoritative; metadata fetches before/after capture
    # must not shorten the trading window. Preserve the original 300s -> 270s rule.
    with gzip.open(directory/'feed.jsonl.gz','rt') as source:
        capture_header=json.loads(next(source))
    duration=float(capture_header['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid capture duration')
    drain_after=max(0.,duration-30.)
    start=None;seq={};last_t=None;draining=False
    seen=set();lags=[]
    with gzip.open(directory/'feed.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);t=row['t'];wall=row['wall'];last_t=t
            if start is None:start=t
            # Last 30 seconds permit delayed risk exits; unfilled leftovers stay unresolved.
            if t-start>=drain_after and not draining:
                draining=True
                for m in markets.values():
                    m.e.drain=True;m.e.request_stop(t)
            if row['kind']=='gap':
                for m in markets.values():m.invalidate(t,'recording_gap')
            data=row.get('data',{});typ=data.get('type');msg=data.get('msg',{})
            sid,s=data.get('sid'),data.get('seq')
            if sid is not None and s is not None:
                if sid in seq and s!=seq[sid]+1:
                    for m in markets.values():m.invalidate(t,'sequence_gap')
                    raise RuntimeError('Sequence gap; refusing a falsely continuous replay')
                seq[sid]=s
            ticker=msg.get('market_ticker');current=markets.get(ticker)
            if current and typ in ('orderbook_snapshot','orderbook_delta'):
                if typ=='orderbook_snapshot':current.raw_book.snapshot(msg,t)
                else:current.raw_book.delta(msg,t)
                if typ=='orderbook_delta' and not current.e.book.ready:
                    # The complete, sequence-checked raw book remains reconstructible after
                    # a strategy staleness pause. Resume from that full contemporaneous state;
                    # never restore old labels or queue priority across the pause.
                    snapshot=dict(market_ticker=ticker,**{side+'_dollars_fp':[
                        [format(p/10000,'.4f'),str(q)] for p,q in current.raw_book.levels[side].items()]
                        for side in ('yes','no')})
                    data=dict(data,type='orderbook_snapshot',msg=snapshot)
                    current.audit['reconstructed_resnapshots_after_pause']+=1
            trade=None
            if typ=='trade' and current and not msg.get('is_block_trade',False):
                tid=msg.get('trade_id')
                if not tid:raise RuntimeError('Missing trade ID')
                if tid not in seen:
                    seen.add(tid)
                    source=float(msg.get('ts_ms',wall*1000))/1000
                    lags.append((wall-source)*1000)
                    trade=dict(t=round(t*NS),side=0 if msg['taker_side']=='yes' else 1,
                               price=round(float(msg['yes_price_dollars'])*10000),qty=float(msg['count_fp']),
                               exchange_t=round((t-max(0,wall-source))*NS))
            for m in markets.values():
                was_valid=m.b.valid
                try:
                    if data:m.feed.message(data,t,wall)
                    else:m.e.tick(t)
                except RuntimeError as exc:
                    m.invalidate(t,type(exc).__name__)
                    raise
                m.sync_book(t)
                if was_valid and not m.b.valid:
                    m.invalidate(t,'strategy_book_pause')
                    m.sync_book(t)
                if wallet.owner==m.e.tag and m.e.order is None and m.e.position is None:
                    wallet.owner=None
                localtrade=trade if m is current else None
                ctx=m.on_trade(localtrade) if localtrade else None
                is_book=m is current and typ in ('orderbook_snapshot','orderbook_delta')
                m.process_quotes(round(t*NS),localtrade,is_book,wallet,draining)
                if is_book and m.b.valid:
                    cost=pair_cost(m.b,1,m.c)
                    m.audit['arbitrage_book_checks']+=1
                    if cost is not None and cost<1:
                        # A valid uncrossed same-contract binary book cannot satisfy this.
                        # Stop rather than book an unverified crossed/stale arbitrage fill.
                        raise RuntimeError('Unexpected paired-taker candidate; verify book before trading')
                    m.audit['arbitrage_no_net_opportunity']+=1
                if ctx:m.decide_maker(round(t*NS),ctx,wallet,draining)
            eq=wallet.cash;known=True
            for m in markets.values():
                if m.active:
                    if m.active.last_mark is None:known=False
                    else:eq+=m.active.last_mark
                p=m.e.position
                if p:
                    fills=m.e.book.fill(p['side'],False,p['q'],m.e.cfg) if m.b.valid else None
                    if fills is None:known=False
                    else:eq+=value(fills)-fee(fills,m.e.multiplier,m.e.cfg)
            if known:
                wallet.peak=max(wallet.peak,eq);wallet.dd=max(wallet.dd,wallet.peak-eq)
                if not wallet.curve or abs(wallet.curve[-1][1]-eq)>1e-9:wallet.curve.append([t,eq])
            else:wallet.unknown_marks+=1
    if incomplete:
        # This is an observation cutoff, not a time at which the strategy knew to
        # stop. Never shorten the planned duration or manufacture a closing fill.
        for m in markets.values():
            for ep in m.e.labels:
                m.e.censor(ep,'recording_end',last_t)
            for sh in m.shadows:
                sh['quote'].censored=True
            if m.active:
                m.active.censored=True
                m.active.reason='recording_end_unresolved'
    result=dict(mode='causal replay of newly live-recorded data; no exchange orders',
                incomplete_capture=incomplete,
                fee_validation='start_and_end' if end_metadata else 'start_only_end_unavailable',
                actual_observed_seconds=last_t-start,
                planned_duration_seconds=duration,
                initial_cash=100.,cash=wallet.cash,max_drawdown_observed=wallet.dd,
                unmarkable_observations=wallet.unknown_marks,markets={},equity=wallet.curve,
                source_lag_ms_median=float(np.median(lags)) if lags else None,
                source_lag_ms_p95=float(np.quantile(lags,.95)) if lags else None)
    for k,m in markets.items():
        ready=[d for d in m.decisions if d['ready']]
        scores={}
        if m.flow_scores:
            for level in ('all','full'):
                scores[level]=dict(n=len(m.flow_scores),next_side_accuracy=float(np.mean([
                    int(np.argmax(r['predictions'][level]['side']))==r['actual_side'] for r in m.flow_scores])),
                    opposite_within_five_accuracy=float(np.mean([
                    (r['predictions'][level]['opp5']>=.5)==r['opp5'] for r in m.flow_scores])))
        result['markets'][k]=dict(title=summary['metadata'][k]['market']['title'],trades=m.e.count,
            maker_multiplier=m.c.maker_multiplier,taker_multiplier=m.c.taker_multiplier,
            taker_history=m.e.history_n,taker_anchors=len(m.e.forecasts),
            taker_closed=len(m.e.trades),taker_gross=m.e.realized_gross,taker_net=m.e.realized_net,
            taker_fees=m.e.paid_fees,taker_directional_accuracy=m.e.stats()['DA_percent'],
            taker_pending_labels=len(m.e.labels),taker_open=bool(m.e.position),
            taker_censored_labels=sum(str(f['label_status']).startswith('censored') for f in m.e.forecasts),
            taker_ready_forecasts=sum(f['ready'] for f in m.e.forecasts),
            taker_chosen_directions=sum(bool(f['side']) and f['ready'] for f in m.e.forecasts),
            maker_closed=len(m.ledger),maker_net=sum(r['net'] for r in m.ledger),
            maker_gross=sum(r['gross'] for r in m.ledger),maker_fees=sum(r['fees'] for r in m.ledger),
            maker_open=bool(m.active),maker_matured_shadow_quotes=m.quote_bank.n,
            trades_in_45_55_band=m.observed_band_trades,
            maker_ready_forecasts=len(ready),maker_qualifying_forecasts=sum(d['qualifies'] for d in ready),
            maker_pending_shadow_quotes=len(m.shadows),flow_matured_windows=m.flow_bank.n,
            flow_accuracy=scores,arbitrage_trades=0,arbitrage_net=0.,audit=dict(m.audit))
        out=directory/k;out.mkdir(exist_ok=True)
        # Individual engine cash/mark traces include shared cash changes. The combined
        # equity curve above, not per-engine drawdown, is the portfolio authority.
        m.e.cash=wallet.cash
        m.e.export(out)
        (out/'maker.json').write_text(json.dumps(dict(decisions=m.decisions,ledger=m.ledger,
             shadows=m.shadow_done,unresolved=m.active.result() if m.active else None),indent=2))
    (directory/'combined_results.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory')
    p.add_argument('--allow-incomplete',action='store_true')
    args=p.parse_args()
    print(json.dumps(run(args.directory,args.allow_incomplete),indent=2))
