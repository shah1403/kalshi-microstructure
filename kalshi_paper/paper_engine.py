"""Local paper ledger only. Network code supports GETs and market-data subscriptions only.
All prices in dollars; counts in unique non-block public trade messages, not volume.
"""
import asyncio
import base64
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING
import gzip
import json
import math
from pathlib import Path
import time
import urllib.parse
import urllib.request

import numpy as np

EPS = 1e-9


@dataclass
class Settings:
    ticker: str = "REPLACE_WITH_MARKET_TICKER"
    bankroll: float = 100.0
    order_budget: float = 5.0
    cash_fraction: float = 0.25
    max_contracts: int = 20
    mode: str = "imbalance_state"
    warmup: int = 30
    min_cell: int = 1  # Expose sparse cells; set 30 for a stricter separate experiment.
    strong_imbalance: float = 0.8
    persistence_trades: int = 3  # New explicit persistence requirement; 1 disables it.
    anchor_stride: int = 5
    max_trades: int = 50
    reference_q: int = 1  # Fixed learning target, independent of the execution size.
    latency_ms: float = 5.0  # Added after local data receipt, on entry AND exit.
    depth_fraction: float = 0.5  # Use only half of displayed depth at each price.
    adverse_slippage: float = 0.001  # Additional adverse dollars/contract per leg.
    fee_quantum: str = "0.0001"  # Current schedule centicent scenario; .01 for old scenario.
    fee_scope: str = "contract"  # Conservative split fills; 'level' aggregates a price level.
    max_book_age_s: float = 10.0
    max_source_lag_s: float = 3.0
    max_processing_lag_s: float = 1.0
    max_hold_s: float = 300.0  # Operational dead-market exit, separately identified.
    close_buffer_s: float = 60.0
    session_loss_limit: float = 10.0
    late_diagnostic_trades: int = 500
    max_pending_labels: int = 2000
    plot_points: int = 2500

    def __post_init__(self):
        assert self.mode in ('imbalance', 'imbalance_state')
        assert self.max_trades == 50 and self.reference_q == 1
        assert self.bankroll > 0 and self.order_budget > 0 and 0 < self.cash_fraction <= 1
        assert self.max_contracts >= 1 and 0 < self.depth_fraction <= 1
        assert self.persistence_trades >= 1 and self.anchor_stride >= 1
        assert self.warmup >= 1 and self.min_cell >= 1 and self.latency_ms >= 0
        assert self.fee_scope in ('contract', 'level') and Decimal(self.fee_quantum) > 0
        assert self.adverse_slippage >= 0 and self.max_hold_s > 0


class Book:
    """Kalshi broadcasts YES bids and NO bids. NO bids imply YES asks at 1-p."""
    def __init__(self):
        self.levels = {'yes': {}, 'no': {}}
        self.last_t = -math.inf
        self.revision = 0
        self.ready = False

    @staticmethod
    def key(p):
        d = Decimal(str(p)) * 10000
        if d != d.to_integral_value() or not 0 < d < 10000:
            raise ValueError('Price outside supported four-decimal binary grid')
        return int(d)

    def snapshot(self, msg, t):
        # Empty sides are omitted by the live feed, including fully empty
        # snapshots when a market's book clears. Always replace old depth.
        # Do not mistake a legacy numeric-price payload for an empty book.
        if any(k in msg for k in ('yes', 'no', 'yes_dollars', 'no_dollars')):
            raise ValueError('Unsupported legacy snapshot fields')
        if any(s+'_dollars_fp' not in msg for s in ('yes','no')) and not msg.get('market_ticker'):
            raise ValueError('Omitted snapshot sides require market identity')
        new = {}
        for side in ('yes', 'no'):
            field = side + '_dollars_fp'
            levels = msg.get(field, [])
            if not isinstance(levels, list):
                raise ValueError('Invalid fixed-point snapshot side: ' + field)
            new[side] = {}
            for p,q in levels:
                q = float(q)
                if not math.isfinite(q) or q < 0:
                    raise ValueError('Invalid snapshot quantity')
                if q > 0:
                    new[side][self.key(p)] = q
        self.levels = new
        self.last_t, self.ready = t, True
        self.revision += 1

    def delta(self, msg, t):
        if not self.ready:
            raise ValueError('Delta arrived before a snapshot')
        side = msg['side']
        if side not in self.levels:
            raise ValueError('Unknown book side')
        key = self.key(msg['price_dollars'])
        q = self.levels[side].get(key, 0.) + float(msg['delta_fp'])
        if not math.isfinite(q) or q < -EPS:
            raise ValueError('Negative or invalid book quantity; resync required')
        if q <= EPS:
            self.levels[side].pop(key, None)
        else:
            self.levels[side][key] = q
        self.last_t = t
        self.revision += 1

    def quote(self):
        y, n = self.levels['yes'], self.levels['no']
        if not self.ready or not y or not n:
            return None
        bid, ask = max(y)/10000, 1-max(n)/10000
        if bid >= ask - EPS:
            return None
        bq, aq = y[max(y)], n[max(n)]
        return dict(bid=bid, ask=ask, mid=(bid+ask)/2,
                    imbalance=(bq-aq)/(bq+aq), spread=ask-bid)

    def fill(self, side, buy, q, cfg):
        if not self.quote() or q < 1:
            return None
        held = 'yes' if side == 1 else 'no'
        source = ('no' if held == 'yes' else 'yes') if buy else held
        left, fills = q, []
        for key, size in sorted(self.levels[source].items(), reverse=True):
            take = min(left, int(math.floor(size*cfg.depth_fraction + EPS)))
            if not take:
                continue
            p = 1-key/10000 if buy else key/10000
            p += cfg.adverse_slippage if buy else -cfg.adverse_slippage
            # Never manufacture a clipped better fill at the binary boundary.
            if not 0 < p < 1:
                return None
            fills.append((take, round(p, 8)))
            left -= take
            if left == 0:
                return fills
        return None  # Simulated marketable fill-or-kill, not a claimed market-order partial fill.


def value(fills):
    return sum(q*p for q, p in fills)


def fee(fills, multiplier, cfg):
    quantum = Decimal(cfg.fee_quantum)
    total = Decimal(0)
    for q, p in fills:
        p = Decimal(str(p))
        unit = Decimal('0.07') * Decimal(str(multiplier)) * p * (1-p)
        groups = q if cfg.fee_scope == 'contract' else 1
        raw = unit if cfg.fee_scope == 'contract' else unit*q
        total += groups * (raw/quantum).to_integral_value(rounding=ROUND_CEILING)*quantum
    return float(total)


def action(probabilities):
    """Order is down, zero, up. No directional imbalance assumption."""
    pm, p0, pp = probabilities
    if pp > max(pm, p0):
        return 1, 1.0, 'normal'
    if pm > max(pp, p0):
        return -1, 1.0, 'normal'
    if p0 > max(pm, pp) and max(pm, pp) > .45 and pm != pp:
        return (1 if pp > pm else -1), .45, 'reduced_45'
    return 0, 0., 'abstain'


def bucket(imbalance):
    return int(np.searchsorted([-.8, -.4, .4, .8], imbalance, side='right'))


class Engine:
    def __init__(self, cfg, multiplier=1.):
        self.cfg, self.multiplier = cfg, float(multiplier)
        self.book = Book()
        self.count = 0
        self.counts = defaultdict(lambda: np.zeros(3, dtype=int))
        self.state, self.state_origin = None, -1
        self.labels, self.forecasts, self.trades, self.events = [], [], [], []
        self.late = []
        self.history_n = 0
        self.cash = cfg.bankroll
        self.position = self.order = None
        self.realized_gross = self.realized_net = self.paid_fees = 0.
        self.peak = cfg.bankroll
        self.max_dd = 0.
        self.unknown_marks = 0
        self.mark = cfg.bankroll
        self.connected = True  # Runner calls gap before connecting.
        self.tradable, self.metadata_t, self.close_t = True, 0., math.inf
        self.paused, self.stop_requested = False, False
        self.run_sign, self.run_count = 0, 0
        self.last_anchor = -10**9
        self.spreads = Counter()
        self.spread_n = 0
        self.points = deque(maxlen=cfg.plot_points)
        self.diagnostics = Counter()
        self.last_probs = np.ones(3)/3
        self.last_cell = None
        self.last_t = 0.

    def event(self, t, kind, **kwargs):
        quote = self.book.quote()
        self.events.append(dict(t=t, kind=kind, trade_count=self.count,
                                mid=quote['mid'] if quote else None, **kwargs))

    def current_key(self, b):
        return (b,) if self.cfg.mode == 'imbalance' else (b, self.state)

    def probabilities(self, b):
        c = self.counts[self.current_key(b)]
        return (c+.5)/(sum(c)+1.5), int(sum(c))

    def quantile_spread(self):
        if self.spread_n < 20:
            return -1.
        h = (self.spread_n-1)*.25
        lo, hi = math.floor(h), math.ceil(h)
        found, acc = {}, 0
        for k, n in sorted(self.spreads.items()):
            for rank in (lo, hi):
                if acc <= rank < acc+n:
                    found[rank] = k/10000
            acc += n
            if len(found) == len(set((lo, hi))):
                break
        return found[lo] + (h-lo)*(found[hi]-found[lo])

    def censor(self, ep, reason, t):
        ep['status'], ep['censor_reason'] = 'censored', reason
        self.forecasts[ep['id']]['label_status'] = 'censored:' + reason
        self.diagnostics['censored:'+reason] += 1
        self.event(t, 'censored', anchor_id=ep['id'], reason=reason)

    def gap(self, t, reason):
        self.last_t = t
        self.connected = False
        self.book.ready = False
        self.run_count, self.run_sign = 0, 0
        self.state, self.state_origin = None, -1
        for ep in self.labels:
            self.censor(ep, reason, t)
        self.labels.clear()
        self.late.clear()
        if self.order:
            self.event(t, 'entry_cancelled_gap', anchor_id=self.order['id'])
            self.order = None
        if self.position:
            self.position['gap_exposed'] = True
            self.request_exit(t, 'data_gap')
        self.event(t, 'data_gap', reason=reason)
        self.mark = None if self.position else self.cash

    def request_exit(self, t, reason):
        if self.position and self.position.get('exit_due') is None:
            self.position.update(exit_due=t+self.cfg.latency_ms/1000,
                                 exit_reason=reason, exit_signal_t=t)
            self.event(t, 'exit_signal', reason=reason, side=self.position['side'])

    def request_stop(self, t):
        self.stop_requested = True
        self.order = None
        self.request_exit(t, 'manual_stop')

    def mature(self, t):
        # Only evidence timestamped strictly before the decision is usable.
        keep = []
        for ep in self.labels:
            if ep.get('y') is not None and ep['hit_t'] < t and ep['id'] > self.state_origin:
                self.state, self.state_origin = ep['y'], ep['id']
            if ep.get('mature_t') is not None and ep['mature_t'] < t:
                y = ep['y']
                for key in [('all',), (ep['bucket'],), (ep['bucket'], ep['state'])]:
                    self.counts[key][y+1] += 1
                self.history_n += 1
                self.forecasts[ep['id']]['label_status'] = 'matured'
                if y == 0 and ep['side']:
                    self.late.append(ep)
            else:
                keep.append(ep)
        self.labels = keep

    def label_update(self, t, is_book):
        for ep in list(self.labels):
            elapsed = self.count-ep['start_count']
            if ep.get('mature_t') is not None:
                continue
            if ep.get('entries') is None:
                if elapsed >= 50:
                    self.censor(ep, 'entry_latency_exceeds_window', t)
                    self.labels.remove(ep)
                    continue
                if not is_book or t < ep['entry_due']:
                    continue
                entries = {s:self.book.fill(s, True, 1, self.cfg) for s in (-1, 1)}
                if any(x is None for x in entries.values()):
                    self.censor(ep, 'reference_depth', t)
                    self.labels.remove(ep)
                    continue
                ep['entries'] = entries
                ep['entry_costs'] = {s:value(f)+fee(f,self.multiplier,self.cfg) for s,f in entries.items()}
                ep['entry_t'] = t
                continue
            if ep.get('y') is None:
                hits = []
                if is_book and t > ep['entry_t']:
                    for side in (-1, 1):
                        out = self.book.fill(side, False, 1, self.cfg)
                        if out and value(out)-fee(out,self.multiplier,self.cfg) > ep['entry_costs'][side]+EPS:
                            hits.append(side)
                if len(hits) == 2:
                    self.censor(ep, 'ambiguous_hit', t)
                    self.labels.remove(ep)
                    continue
                if hits:
                    ep.update(y=hits[0], hit_t=t)
                    f = self.forecasts[ep['id']]
                    f.update(y=hits[0], hit_t=t, hit_wait_trades=elapsed, hit_wait_s=t-ep['t'])
                    kind = ('right_move' if ep['side'] == hits[0] else 'wrong_move') if f['ready'] and ep['side'] else 'untraded_move'
                    self.event(t, kind, anchor_id=ep['id'], side=hits[0])
                elif elapsed >= 50:
                    ep.update(y=0, hit_t=t)
                    self.forecasts[ep['id']].update(y=0, hit_t=t)
                    self.event(t, 'label_timeout', anchor_id=ep['id'])
            if elapsed >= 50:
                ep['mature_t'] = t
        if is_book:
            for ep in list(self.late):
                wait = self.count-ep['start_count']
                if wait > self.cfg.late_diagnostic_trades:
                    self.late.remove(ep)
                    continue
                out = self.book.fill(ep['side'], False, 1, self.cfg)
                if out and value(out)-fee(out,self.multiplier,self.cfg) > ep['entry_costs'][ep['side']]+EPS:
                    self.forecasts[ep['id']].update(late_same_side_wait=wait, late_same_side_seconds=t-ep['t'])
                    self.event(t, 'late_same_side', anchor_id=ep['id'])
                    self.late.remove(ep)

    def size(self, probs, side, scale):
        p = probs[2 if side == 1 else 0]
        budget = min(self.cfg.order_budget, self.cash*self.cfg.cash_fraction)*p
        normal_q = 0
        for q in range(1, self.cfg.max_contracts+1):
            fills = self.book.fill(side, True, q, self.cfg)
            if fills is None or value(fills)+fee(fills,self.multiplier,self.cfg) > min(self.cash,budget)+EPS:
                break
            normal_q = q
        return math.floor(normal_q*scale+EPS), normal_q

    def anchor(self, t, quote):
        b = bucket(quote['imbalance'])
        p, n = self.probabilities(b)
        side, scale, tier = action(p)
        key = self.current_key(b)
        ready = self.history_n >= self.cfg.warmup and n >= self.cfg.min_cell
        q, normal = self.size(p, side, scale) if side and ready else (0,0)
        idx = len(self.forecasts)
        f = dict(id=idx, t=t, start_count=self.count, bucket=b, state=self.state,
                 pminus=float(p[0]), pzero=float(p[1]), pplus=float(p[2]),
                 cell_n=n, history_n=self.history_n, ready=ready, side=side, tier=tier,
                 q_requested=q, normal_q=normal, taken=False, y=None, label_status='pending')
        self.forecasts.append(f)
        ep = dict(id=idx,t=t,start_count=self.count,bucket=b,state=self.state,side=side,
                  entry_due=t+self.cfg.latency_ms/1000,status='pending')
        self.labels.append(ep)
        self.last_probs, self.last_cell = p, key
        self.event(t, 'anchor', anchor_id=idx, side=side, tier=tier, cell_n=n)
        if side and ready and q == 0:
            self.diagnostics['size_below_one'] += 1
        if q and self.position is None and self.order is None and not self.paused:
            self.order = dict(id=idx,side=side,q=q,t=t,due=ep['entry_due'],
                              start_count=self.count,tier=tier)

    def execution(self, t, is_book):
        if self.order and (self.count-self.order['start_count'] >= 50 or self.stop_requested):
            self.event(t, 'entry_expired', anchor_id=self.order['id'])
            self.order = None
        if is_book and self.order and t >= self.order['due']:
            o, self.order = self.order, None
            ent = self.book.fill(o['side'], True, o['q'], self.cfg)
            if ent is None or value(ent)+fee(ent,self.multiplier,self.cfg) > self.cash+EPS:
                self.event(t, 'entry_rejected', anchor_id=o['id'])
            else:
                ef = fee(ent,self.multiplier,self.cfg)
                self.cash -= value(ent)+ef
                self.paid_fees += ef
                self.position = dict(**o, entry=ent, entry_t=t, entry_fee=ef,
                                     exit_due=None, gap_exposed=False, retry_rev=-1)
                self.forecasts[o['id']]['taken'] = True
                self.event(t, 'entry', anchor_id=o['id'], side=o['side'],q=o['q'],
                           price=value(ent)/o['q'],entry_delay_ms=(t-o['t'])*1000)
        pos = self.position
        if pos:
            out = self.book.fill(pos['side'], False, pos['q'], self.cfg)
            if is_book and t > pos['entry_t'] and out:
                net = value(out)-value(pos['entry'])-pos['entry_fee']-fee(out,self.multiplier,self.cfg)
                if net > EPS:
                    self.request_exit(t, 'profit_opportunity')
            if self.count-pos['start_count'] >= 50:
                self.request_exit(t, '50_trade_timeout')
            if t-pos['entry_t'] >= self.cfg.max_hold_s:
                self.request_exit(t, 'wall_clock_safety_exit')
            if is_book and pos.get('exit_due') is not None and t >= pos['exit_due'] and t > pos['exit_signal_t']:
                if out:
                    xf = fee(out,self.multiplier,self.cfg)
                    gross = value(out)-value(pos['entry'])
                    net = gross-pos['entry_fee']-xf
                    self.cash += value(out)-xf
                    self.paid_fees += xf
                    self.realized_gross += gross
                    self.realized_net += net
                    tr = dict(**pos, exit=out,exit_t=t,gross=gross,fees=pos['entry_fee']+xf,
                              net=net,exit_delay_ms=(t-pos['exit_signal_t'])*1000,
                              wait_trades=self.count-pos['start_count'])
                    self.trades.append(tr)
                    self.event(t, 'exit',anchor_id=pos['id'],side=pos['side'],net=net,
                               reason=pos['exit_reason'],price=value(out)/pos['q'])
                    self.position = None
                else:
                    # Retry only at a later fresh book and after another configured latency.
                    pos['exit_due'] = t+self.cfg.latency_ms/1000
                    self.diagnostics['exit_depth_rejections'] += 1

    def marking(self, t):
        if self.position:
            out = self.book.fill(self.position['side'],False,self.position['q'],self.cfg)
            self.mark = self.cash+value(out)-fee(out,self.multiplier,self.cfg) if out else None
        else:
            self.mark = self.cash
        if self.mark is None:
            self.unknown_marks += 1
        else:
            self.peak = max(self.peak,self.mark)
            self.max_dd = max(self.max_dd,self.peak-self.mark)
            if self.cfg.bankroll-self.mark >= self.cfg.session_loss_limit:
                self.paused = True
                self.order = None
                self.request_exit(t,'loss_limit')

    def tick(self, t, is_book=False, is_trade=False):
        self.last_t = t
        if is_trade:
            self.count += 1  # Even if current book is invalid: never compress the trade clock.
        if not self.tradable:
            self.order = None
            for ep in self.labels:
                self.censor(ep,'market_not_tradable',t)
            self.labels.clear()
            self.mark = None if self.position else self.cash
            return  # Never value a settlement or terminal book as an executable fill.
        quote = self.book.quote()
        if not self.connected or not quote:
            if self.connected:
                self.gap(t,'invalid_book')
            return
        if t-self.book.last_t > self.cfg.max_book_age_s:
            self.gap(t,'stale_book')
            return
        if is_book:
            current_bucket=bucket(quote['imbalance'])
            for ep in self.labels:
                if current_bucket!=ep['bucket'] and not ep.get('imbalance_changed'):
                    ep['imbalance_changed']=True
                    self.forecasts[ep['id']]['imbalance_changed']=True
                    self.event(t,'anchor_imbalance_changed',anchor_id=ep['id'])
        if t >= self.close_t-self.cfg.close_buffer_s or not self.tradable or self.stop_requested:
            self.order = None
            self.request_exit(t,'market_closing' if not self.stop_requested else 'manual_stop')
        self.mature(t)
        self.label_update(t,is_book)
        self.execution(t,is_book)
        self.marking(t)
        if is_trade:
            im = quote['imbalance']
            sg = int(np.sign(im)) if abs(im) >= self.cfg.strong_imbalance else 0
            if sg and sg == self.run_sign:
                self.run_count += 1
            else:
                if self.run_count >= self.cfg.persistence_trades:
                    self.event(t,'imbalance_unlocked')
                self.run_count = 1 if sg else 0
                self.run_sign = sg
            if self.run_count == self.cfg.persistence_trades:
                self.event(t,'imbalance_locked',imbalance=im)
            spread_key = round(quote['spread']*10000)
            self.spreads[spread_key] += 1
            self.spread_n += 1
            allowed = (self.tradable and not self.stop_requested and not self.paused
                       and t < self.close_t-self.cfg.close_buffer_s)
            if (allowed and self.run_count >= self.cfg.persistence_trades
                    and self.count-self.last_anchor >= self.cfg.anchor_stride
                    and quote['spread'] <= self.quantile_spread()+EPS):
                if len(self.labels) < self.cfg.max_pending_labels:
                    self.anchor(t,quote)
                    self.last_anchor = self.count
                else:
                    self.diagnostics['pending_limit'] += 1
        if is_book or is_trade:
            self.points.append(dict(t=t,**quote,net_equity=None if self.mark is None else self.mark-self.cfg.bankroll,
                                    realized_gross=self.realized_gross,realized_net=self.realized_net))

    def stats(self):
        valid = [f for f in self.forecasts if f['ready'] and f['label_status']=='matured']
        directional = [f for f in valid if f['side'] and f['y'] in (-1,1)]
        taken = [f for f in directional if f['taken']]
        all_signals = [f for f in valid if f['side']]
        def da(rows):
            return 100*sum(f['side']==f['y'] for f in rows)/len(rows) if rows else None
        return dict(trade_messages=self.count,history_n=self.history_n,anchors=len(self.forecasts),
                    cell_n=int(sum(self.counts[self.last_cell])) if self.last_cell else 0,
                    pending_labels=len(self.labels),closed_trades=len(self.trades),
                    open_contracts=self.position['q'] if self.position else 0,
                    gross_closed=self.realized_gross,fees_closed=sum(x['fees'] for x in self.trades),
                    net_closed=self.realized_net,fees_paid_including_open=self.paid_fees,
                    cash=self.cash,liquidation_equity=self.mark,max_dd_observed=self.max_dd,
                    unmarkable_observations=self.unknown_marks,
                    DA_percent=da(directional),DA_n=len(directional),traded_DA_percent=da(taken),traded_DA_n=len(taken),
                    directional_signals_with_valid_labels=len(all_signals),
                    correct_direction_hit_rate=100*sum(f['side']==f['y'] for f in all_signals)/len(all_signals) if all_signals else None,
                    directional_signal_timeouts=sum(f['y']==0 for f in all_signals),
                    taken_censored=sum(f['taken'] and f['label_status'].startswith('censored') for f in self.forecasts),
                    taken_pending=sum(f['taken'] and f['label_status']=='pending' for f in self.forecasts),
                    win_rate=100*sum(x['net']>0 for x in self.trades)/len(self.trades) if self.trades else None,
                    timeouts=sum(x['exit_reason']=='50_trade_timeout' for x in self.trades),
                    late_same_side=sum('late_same_side_wait' in f for f in self.forecasts),
                    connected=self.connected,paused=self.paused,**dict(self.diagnostics))

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        def default(x):
            if isinstance(x,np.generic): return x.item()
            if isinstance(x,np.ndarray): return x.tolist()
            raise TypeError(type(x).__name__)
        out = dict(config=asdict(self.cfg),fee_multiplier=self.multiplier,stats=self.stats(),
                   forecasts=self.forecasts,trades=self.trades,events=self.events,
                   open_position=self.position,pending_order=self.order,
                   counts={str(k):v.tolist() for k,v in self.counts.items()})
        (directory/'ledger.json').write_text(json.dumps(out,indent=2,default=default),encoding='utf-8')
        import pandas as pd
        for name in ('forecasts','trades','events'):
            pd.DataFrame(getattr(self,name)).to_csv(directory/(name+'.csv'),index=False)


REST = 'https://external-api.kalshi.com/trade-api/v2'
WS = 'wss://external-api-ws.kalshi.com/trade-api/ws/v2'


def public_get(path, params=None):
    # This function cannot issue portfolio requests or any POST/DELETE/PATCH.
    if not path.startswith(('/markets','/events/','/series/')):
        raise ValueError('Only public market-data GET endpoints are allowed')
    url = REST+path
    if params:
        url += '?'+urllib.parse.urlencode(params)
    request = urllib.request.Request(url,headers={'User-Agent':'local-paper-research/1.0'},method='GET')
    with urllib.request.urlopen(request,timeout=15) as response:
        return json.load(response)


def list_markets(series_ticker='', pages=3):
    """Bounded listing, not a claim to scan the entire exchange."""
    import pandas as pd
    rows, cursor = [], ''
    for _ in range(pages):
        params = dict(status='open',limit=200)
        if series_ticker: params['series_ticker']=series_ticker
        if cursor: params['cursor']=cursor
        response = public_get('/markets',params)
        rows.extend(response.get('markets',[]))
        cursor = response.get('cursor','')
        if not cursor: break
    fields = ['ticker','title','status','yes_bid_dollars','yes_ask_dollars','volume_24h_fp','close_time']
    frame = pd.DataFrame(rows).reindex(columns=fields)
    if len(frame):
        frame['volume_24h_fp'] = pd.to_numeric(frame['volume_24h_fp'],errors='coerce')
        frame = frame.sort_values('volume_24h_fp',ascending=False)
    return frame


def market_metadata(ticker):
    market = public_get('/markets/'+urllib.parse.quote(ticker,safe=''))['market']
    event = public_get('/events/'+urllib.parse.quote(market['event_ticker'],safe=''))['event']
    series = public_get('/series/'+urllib.parse.quote(event['series_ticker'],safe=''))['series']
    series = dict(series)
    for field in ('fee_type','fee_multiplier'):
        override = event.get(field+'_override')
        if override is not None:
            series[field] = override
    series['resolved_event_ticker'] = market['event_ticker']
    if series.get('fee_type') not in ('quadratic','quadratic_with_maker_fees'):
        raise ValueError('Unsupported fee type; inspect official fee schedule: '+str(series.get('fee_type')))
    multiplier = float(series['fee_multiplier'])
    if not math.isfinite(multiplier) or multiplier < 0:
        raise ValueError('Invalid fee multiplier')
    if market.get('market_type','binary') != 'binary':
        raise ValueError('Only binary markets supported')
    return market,series


def auth_headers(key_id, private_key):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    stamp = str(int(time.time()*1000))
    text = (stamp+'GET'+'/trade-api/ws/v2').encode()
    signature = private_key.sign(text,padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                                 salt_length=padding.PSS.DIGEST_LENGTH),hashes.SHA256())
    return {'KALSHI-ACCESS-KEY':key_id,'KALSHI-ACCESS-TIMESTAMP':stamp,
            'KALSHI-ACCESS-SIGNATURE':base64.b64encode(signature).decode()}


class Feed:
    """One selected market, per-subscription sequence checks and trade-ID deduplication."""
    def __init__(self, engine):
        self.engine = engine
        self.seq = {}
        self.seen = set()
        self.ids = deque()
        self.event_ticker = None
        self.metadata_dirty = False

    def reset(self):
        self.seq.clear()

    def message(self, data, t, wall):
        e = self.engine
        kind, msg = data.get('type'), data.get('msg',{})
        if kind == 'error':
            raise RuntimeError('Kalshi subscription error: '+str(msg.get('code')))
        if kind not in ('orderbook_snapshot','orderbook_delta','trade',
                        'market_lifecycle_v2','event_lifecycle','event_fee_update'):
            return
        sid, seq = data.get('sid'),data.get('seq')
        if seq is not None and sid is not None:
            previous = self.seq.get(sid)
            if previous is not None and int(seq) != previous+1:
                raise RuntimeError('Sequence discontinuity')
            self.seq[sid] = int(seq)
        if kind == 'event_fee_update' and msg.get('event_ticker') == self.event_ticker:
            e.paused=True
            e.gap(t,'fee_update_start_new_session')
            self.metadata_dirty=True
            return
        if kind == 'market_lifecycle_v2' and msg.get('market_ticker') == e.cfg.ticker:
            event = msg.get('event_type')
            if msg.get('close_ts') is not None:
                e.close_t = t+float(msg['close_ts'])-wall
            if event in ('deactivated','determined','settled'):
                e.tradable=False
                e.tick(t)
            self.metadata_dirty=True
            return
        if kind in ('event_lifecycle','event_fee_update','market_lifecycle_v2'):
            return
        if msg.get('market_ticker') != e.cfg.ticker:
            return
        source = msg.get('ts_ms')
        if source is not None and wall-float(source)/1000 > e.cfg.max_source_lag_s:
            raise RuntimeError('Source timestamp is stale; check local clock and feed')
        if kind == 'trade':
            tid = msg.get('trade_id')
            if not tid:
                raise RuntimeError('Trade ID missing; cannot count unique trades')
            if tid in self.seen or msg.get('is_block_trade',False):
                return
            self.seen.add(tid)
            self.ids.append(tid)
            if len(self.ids)>100000:
                self.seen.remove(self.ids.popleft())
            e.tick(t,is_trade=True)
        else:
            if kind == 'orderbook_snapshot':
                if e.connected and e.book.ready:
                    e.gap(t,'unsolicited_snapshot')
                e.book.snapshot(msg,t)
                e.connected=True
            else:
                e.book.delta(msg,t)
            e.tick(t,is_book=True)


class PaperSession:
    def __init__(self,cfg,key_id,private_key,output='paper_run'):
        self.cfg,self.key_id,self.private_key=cfg,key_id,private_key
        self.engine=Engine(cfg)
        self.feed=Feed(self.engine)
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.error=None
        self.task=None
        self.hard_stop=False
        self.log=None
        self.receipt_lags=deque(maxlen=1000)

    def record(self, kind, t, **payload):
        if self.log:
            self.log.write(json.dumps(dict(kind=kind,t=t,wall=time.time(),**payload))+'\n')
            self.log.flush()

    async def refresh_metadata(self):
        market,series=await asyncio.to_thread(market_metadata,self.cfg.ticker)
        e=self.engine
        t=time.monotonic()
        multiplier=float(series['fee_multiplier'])
        if e.history_n and multiplier != e.multiplier:
            e.paused=True
            e.gap(t,'fee_change_start_new_session')
        e.multiplier=multiplier
        from datetime import datetime
        close=datetime.fromisoformat(market['close_time'].replace('Z','+00:00')).timestamp()
        e.close_t=t+close-time.time()
        e.tradable=market['status'] in ('active','open')
        e.metadata_t=t
        self.feed.event_ticker=market['event_ticker']
        self.feed.metadata_dirty=False
        self.record('metadata',t,market=market,series=series,close_t=e.close_t)
        return market,series

    async def run(self):
        import websockets
        e=self.engine
        # Refuse to append a second session to an existing recording.
        self.log=gzip.open(self.output/'feed.jsonl.gz','xt',encoding='utf-8')
        self.record('config',time.monotonic(),config=asdict(self.cfg))
        try:
            await self.refresh_metadata()
            if not e.tradable:
                raise RuntimeError('Selected market is not currently tradable')
            backoff=1
            while not self.hard_stop:
                try:
                    self.feed.reset()
                    e.gap(time.monotonic(),'connection_start')
                    self.record('gap',e.last_t,reason='connection_start')
                    async with websockets.connect(WS,additional_headers=auth_headers(self.key_id,self.private_key),
                                                  ping_interval=10,ping_timeout=10,max_queue=32) as ws:
                        connect_t=time.monotonic()
                        initial_revision=e.book.revision
                        for idx,channel in enumerate(('orderbook_delta','trade'),1):
                            await ws.send(json.dumps({'id':idx,'cmd':'subscribe','params':{
                                'channels':[channel],'market_tickers':[self.cfg.ticker]}}))
                        await ws.send(json.dumps({'id':3,'cmd':'subscribe','params':{
                            'channels':['market_lifecycle_v2']}}))
                        backoff=1
                        while not self.hard_stop:
                            if e.stop_requested and e.position is None:
                                self.hard_stop=True
                                break
                            try:
                                raw=await asyncio.wait_for(ws.recv(),timeout=.1)
                                t,wall=time.monotonic(),time.time()
                                data=json.loads(raw)
                                ts_ms=data.get('msg',{}).get('ts_ms')
                                if ts_ms is not None:
                                    self.receipt_lags.append(max(0,1000*wall-float(ts_ms)))
                                self.record('message',t,data=data)
                                self.feed.message(data,t,wall)
                                if time.monotonic()-t>self.cfg.max_processing_lag_s:
                                    raise RuntimeError('Local processing too slow for feed')
                            except asyncio.TimeoutError:
                                t=time.monotonic()
                                self.record('timer',t)
                                e.tick(t)
                            if not e.connected and (e.book.revision>initial_revision or time.monotonic()-connect_t>10):
                                raise RuntimeError('Book invalid or stale; reconnecting for snapshot')
                            if self.feed.metadata_dirty or time.monotonic()-e.metadata_t>30:
                                await self.refresh_metadata()
                            if not e.tradable:
                                e.request_stop(time.monotonic())
                                self.error='Market halted/closed; remaining inventory is unresolved. Settlement ignored.'
                                self.hard_stop=True
                            if e.stop_requested and e.position and time.monotonic()-e.position.get('exit_signal_t',time.monotonic())>30:
                                self.error='Stopped with unresolved inventory; no settlement PnL booked.'
                                self.hard_stop=True
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.error=type(exc).__name__+': '+str(exc)[:180]
                    t=time.monotonic()
                    e.gap(t,'feed_or_metadata_error')
                    self.record('gap',t,reason='feed_or_metadata_error')
                    e.export(self.output)
                    if e.stop_requested:
                        break
                    await asyncio.sleep(backoff)
                    backoff=min(30,backoff*2)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error=type(exc).__name__+': '+str(exc)[:180]
        finally:
            t=time.monotonic()
            e.gap(t,'session_ended')
            self.record('gap',t,reason='session_ended')
            e.export(self.output)
            self.log.close()
            self.log=None

    def start(self):
        if self.task and not self.task.done():
            raise RuntimeError('Session already running')
        self.task=asyncio.create_task(self.run())
        return self.task

    async def stop(self,grace_seconds=35):
        t=time.monotonic()
        self.record('stop',t)
        self.engine.request_stop(t)
        if self.task:
            try:
                await asyncio.wait_for(asyncio.shield(self.task),timeout=grace_seconds)
            except asyncio.TimeoutError:
                self.hard_stop=True
                self.task.cancel()
                try: await self.task
                except asyncio.CancelledError: pass
        return self.engine.stats()


def replay_recording(path):
    """Same engine, original local receipt ordering. Never sorts by future exchange timestamps."""
    e=feed=None
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line); t=r['t']; kind=r['kind']
            if kind=='config':
                if e is not None: raise ValueError('Use one recorded session per file')
                e=Engine(Settings(**r['config'])); feed=Feed(e)
            elif kind=='metadata':
                m,s=r['market'],r['series']
                multiplier=float(s['fee_multiplier'])
                if e.history_n and multiplier!=e.multiplier:
                    e.paused=True; e.gap(t,'fee_change_start_new_session')
                e.multiplier=multiplier; e.close_t=r['close_t']
                e.tradable=m['status'] in ('active','open'); e.metadata_t=t
                feed.event_ticker=m['event_ticker']; feed.metadata_dirty=False
            elif kind=='gap':
                e.gap(t,r['reason']); feed.reset()
            elif kind=='stop': e.request_stop(t)
            elif kind=='timer': e.tick(t)
            elif kind=='message':
                try: feed.message(r['data'],t,r['wall'])
                except (ValueError,RuntimeError): e.gap(t,'feed_or_metadata_error')
    return e


def dashboard(engine):
    import matplotlib.pyplot as plt
    e=engine
    fig,axes=plt.subplots(3,2,figsize=(16,12),constrained_layout=True)
    pts=list(e.points)
    if not pts:
        axes[0,0].text(.1,.5,'Waiting for a valid orderbook...')
        return fig
    t0=pts[0]['t']
    x=[p['t']-t0 for p in pts]
    ax=axes[0,0]
    for name,color in [('bid','#3b82f6'),('mid','#222222'),('ask','#f59e0b')]:
        ax.plot(x,[p[name] for p in pts],label=name,color=color,lw=1)
    styles={'imbalance_locked':('s','#8b5cf6'),'entry':('^','#007f7f'),
            'right_move':('o','#16a34a'),'wrong_move':('X','#dc2626'),
            'label_timeout':('|','#f59e0b'),'exit':('x','#111111'),'data_gap':('D','#999999')}
    for kind,(marker,color) in styles.items():
        events=[v for v in e.events if v['kind']==kind and v['t']>=t0 and v.get('mid') is not None]
        if events:
            if kind=='entry':
                for side,mk in [(1,'^'),(-1,'v')]:
                    z=[v for v in events if v['side']==side]
                    ax.scatter([v['t']-t0 for v in z],[v['mid'] for v in z],marker=mk,color=color,s=80,label='entry UP' if side==1 else 'entry DOWN')
            elif kind=='exit':
                for reason,label,col in [('50_trade_timeout','timeout exit','#f59e0b'),('profit_opportunity','movement exit','#111111')]:
                    z=[v for v in events if v['reason']==reason]
                    ax.scatter([v['t']-t0 for v in z],[v['mid'] for v in z],marker='x',color=col,s=75,label=label)
                z=[v for v in events if v['reason'] not in ('50_trade_timeout','profit_opportunity')]
                ax.scatter([v['t']-t0 for v in z],[v['mid'] for v in z],marker='x',color='#7c3aed',s=75,label='safety / manual exit')
            else:
                ax.scatter([v['t']-t0 for v in events],[v['mid'] for v in events],marker=marker,color=color,s=40,label=kind)
    ax.set(title='YES bid / mid / ask; direction marks use YES-price coordinates',ylabel='Dollars')
    ax.legend(fontsize=7,ncol=3)
    axes[0,1].plot(x,[p['imbalance'] for p in pts],lw=1)
    for level in [-e.cfg.strong_imbalance,e.cfg.strong_imbalance]: axes[0,1].axhline(level,ls='--',color='grey')
    locks=[v for v in e.events if v['kind']=='imbalance_locked' and v['t']>=t0]
    axes[0,1].scatter([v['t']-t0 for v in locks],[v['imbalance'] for v in locks],color='#8b5cf6',marker='s')
    axes[0,1].set(title='Top-of-book imbalance and persistent locks',ylim=(-1.05,1.05))
    for name,color in [('realized_gross','#3b82f6'),('realized_net','#16a34a'),('net_equity','#9333ea')]:
        axes[1,0].plot(x,[p[name] for p in pts],label=name,color=color)
    axes[1,0].legend(fontsize=8); axes[1,0].set(title='Gross, net closed, and liquidation PnL',ylabel='USD')
    allc=e.counts[('all',)]
    axes[1,1].bar(np.arange(3)-.18,(allc+.5)/(sum(allc)+1.5),width=.36,label=f'All matured n={sum(allc)}')
    axes[1,1].bar(np.arange(3)+.18,e.last_probs,width=.36,label=f'Latest conditioned cell {e.last_cell}')
    axes[1,1].set(xticks=range(3),xticklabels=['DOWN','ZERO','UP'],ylim=(0,1),title='Smoothed empirical first-passage distributions')
    axes[1,1].legend(fontsize=8)
    keys=[k for k in e.counts if k!=('all',) and len(k)==(1 if e.cfg.mode=='imbalance' else 2)]
    if keys:
        keys=sorted(keys,key=str)
        mat=np.array([(e.counts[k]+.5)/(sum(e.counts[k])+1.5) for k in keys])
        axes[2,0].imshow(mat,aspect='auto',vmin=0,vmax=1,cmap='Blues')
        for row in range(len(keys)):
            for col in range(3):
                axes[2,0].text(col,row,f'{mat[row,col]:.2f}',ha='center',va='center',
                               color='white' if mat[row,col]>.55 else '#222222',fontsize=8)
        axes[2,0].set(yticks=range(len(keys)),yticklabels=[f'{k} n={sum(e.counts[k])}' for k in keys],
                      xticks=range(3),xticklabels=['DOWN','ZERO','UP'],title='Conditional cells: (imbalance bucket, observable state)')
    moves=[f for f in e.forecasts if f.get('hit_wait_trades') is not None]
    if moves: axes[2,1].hist([f['hit_wait_trades'] for f in moves],bins=np.arange(0,56,5))
    axes[2,1].set(title='Observed fee-covering first-passage wait',xlabel='Actual trade messages',ylabel='Anchors (overlapping)')
    for ax in [axes[0,0],axes[0,1],axes[1,0]]: ax.set_xlabel('Seconds within displayed window')
    fig.suptitle(f'PAPER ONLY | {e.cfg.ticker} | {e.cfg.mode} | no settlement PnL',fontsize=14)
    return fig
