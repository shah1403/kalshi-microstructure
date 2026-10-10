"""Causal imbalance-threshold diagnostic on a completed Kalshi capture.

Every variant uses the same executable, fee-covering first-passage outcome and
50-trade maturity. Direction for this diagnostic is only sign(imbalance).
No funded trades are created.
"""
import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'kalshi_paper'))
from paper_engine import Book as RawBook, Engine, Feed, Settings, bucket


class DiagnosticEngine(Engine):
    def size(self, probabilities, side, scale):
        return 0, 0

    def anchor(self, t, quote):
        before = len(self.forecasts)
        b = bucket(quote['imbalance'])
        global_counts = self.counts[('all',)].copy()
        imbalance_counts = self.counts[(b,)].copy()
        global_p = (global_counts + .5) / (sum(global_counts) + 1.5)
        imbalance_p = (imbalance_counts + .5) / (sum(imbalance_counts) + 1.5)
        super().anchor(t, quote)
        if len(self.forecasts) == before:
            return
        im = float(quote['imbalance'])
        self.forecasts[-1].update(anchor_imbalance=im,
                                  anchor_sign=int(np.sign(im)),
                                  global_p=global_p.tolist(),
                                  imbalance_p=imbalance_p.tolist(),
                                  weakened_below_08=False,
                                  crossed_hysteresis_04=False,
                                  sign_flip=False,
                                  strong_sign_flip=False)
        self.labels[-1].update(anchor_imbalance=im,
                               anchor_sign=int(np.sign(im)))

    def tick(self, t, is_book=False, is_trade=False):
        # The book has already absorbed this update. Record signal decay before
        # label_update decides whether this same update creates first passage.
        # If it does, restore the prior flags so "before" excludes the hit update.
        quote = self.book.quote()
        touched = []
        if is_book and self.connected and quote:
            im = float(quote['imbalance'])
            sign = int(np.sign(im))
            for ep in self.labels:
                if ep.get('y') is not None or 'anchor_sign' not in ep:
                    continue
                f = self.forecasts[ep['id']]
                old = {k: f[k] for k in ('weakened_below_08',
                    'crossed_hysteresis_04', 'sign_flip', 'strong_sign_flip')}
                touched.append((ep, old))
                anchor_sign = ep['anchor_sign']
                if abs(im) < .8:
                    f['weakened_below_08'] = True
                if anchor_sign * im < .4:
                    f['crossed_hysteresis_04'] = True
                if sign and sign != anchor_sign:
                    f['sign_flip'] = True
                if abs(im) >= .8 and sign != anchor_sign:
                    f['strong_sign_flip'] = True
        super().tick(t, is_book, is_trade)
        for ep, old in touched:
            if ep.get('y') is not None:
                self.forecasts[ep['id']].update(old)


def summarize(engine):
    rows = [f for f in engine.forecasts
            if f['label_status'] == 'matured' and f.get('y') in (-1, 0, 1)]
    same = sum(f['y'] == f['anchor_sign'] for f in rows)
    opposite = sum(f['y'] == -f['anchor_sign'] for f in rows)
    zero = sum(f['y'] == 0 for f in rows)
    moves = same + opposite
    def subset(flag, value=True):
        x = [f for f in rows if bool(f.get(flag)) == value]
        s = sum(f['y'] == f['anchor_sign'] for f in x)
        o = sum(f['y'] == -f['anchor_sign'] for f in x)
        z = sum(f['y'] == 0 for f in x)
        return dict(n=len(x), same=s, opposite=o, zero=z,
                    p_same=s/len(x) if x else None,
                    p_opposite=o/len(x) if x else None,
                    p_zero=z/len(x) if x else None,
                    direction_accuracy_given_move=s/(s+o) if s+o else None)
    return dict(anchors=len(engine.forecasts), matured=len(rows),
                same=same, opposite=opposite, zero=zero,
                p_same=same/len(rows) if rows else None,
                p_opposite=opposite/len(rows) if rows else None,
                p_zero=zero/len(rows) if rows else None,
                direction_accuracy_given_move=same/moves if moves else None,
                weakened_below_08=subset('weakened_below_08'),
                stayed_at_or_above_08=subset('weakened_below_08', False),
                crossed_hysteresis_04=subset('crossed_hysteresis_04'),
                stayed_inside_hysteresis_04=subset('crossed_hysteresis_04', False),
                sign_flip=subset('sign_flip'), strong_sign_flip=subset('strong_sign_flip'))


def row_stats(rows):
    same = sum(f['y'] == f['anchor_sign'] for f in rows)
    opposite = sum(f['y'] == -f['anchor_sign'] for f in rows)
    zero = sum(f['y'] == 0 for f in rows)
    moves = same + opposite
    return dict(n=len(rows), same=same, opposite=opposite, zero=zero,
                p_same=same/len(rows) if rows else None,
                p_opposite=opposite/len(rows) if rows else None,
                p_zero=zero/len(rows) if rows else None,
                direction_accuracy_given_move=same/moves if moves else None)


def learned_stats(rows):
    rows = [f for f in rows if f.get('ready')]
    moves = [f for f in rows if f['y'] in (-1, 1)]

    def score(name, probs):
        predicted = []
        correct = 0
        log_losses = []
        agrees_with_imbalance = 0
        for f in moves:
            pm, _, pp = probs(f)
            if abs(pm-pp) <= 1e-15:
                continue
            side = 1 if pp > pm else -1
            predicted.append(side)
            correct += side == f['y']
            agrees_with_imbalance += side == f['anchor_sign']
            denom = pm + pp
            probability = (pp if f['y'] == 1 else pm) / denom
            log_losses.append(-math.log(max(probability, 1e-15)))
        return dict(name=name, eligible_ready=len(rows), actual_moves=len(moves),
                    directional_predictions=len(predicted), ties=len(moves)-len(predicted),
                    direction_accuracy=correct/len(predicted) if predicted else None,
                    direction_log_loss=float(np.mean(log_losses)) if log_losses else None,
                    agreement_with_initial_imbalance=(agrees_with_imbalance/len(predicted)
                                                      if predicted else None))

    raw_correct = sum(f['anchor_sign'] == f['y'] for f in moves)
    action_rows = [f for f in rows if f['side']]
    action_moves = [f for f in action_rows if f['y'] in (-1, 1)]
    return dict(
        raw_imbalance_sign=dict(eligible_ready=len(rows), actual_moves=len(moves),
            direction_accuracy=raw_correct/len(moves) if moves else None),
        global_history=score('global_history', lambda f: f['global_p']),
        imbalance_only=score('imbalance_only', lambda f: f['imbalance_p']),
        imbalance_and_previous_state=score('imbalance_and_previous_state',
            lambda f: (f['pminus'], f['pzero'], f['pplus'])),
        actual_action_rule=dict(signals=len(action_rows), signals_with_move=len(action_moves),
            correct_direction=sum(f['side'] == f['y'] for f in action_moves),
            zero_outcomes=sum(f['y'] == 0 for f in action_rows),
            direction_accuracy_given_move=(sum(f['side'] == f['y'] for f in action_moves)
                                           / len(action_moves) if action_moves else None)))


def run(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'capture_summary.json').read_text())
    variants = {
        'all_nonzero_no_persistence': (0.0, 1),
        'large_08_no_persistence': (0.8, 1),
        'large_08_persistent_3': (0.8, 3),
    }
    engines = {}
    raw_books = {ticker: RawBook() for ticker in summary['metadata']}
    for name, (threshold, persistence) in variants.items():
        engines[name] = {}
        for ticker, meta in summary['metadata'].items():
            multiplier = float(meta['series']['fee_multiplier'])
            cfg = Settings(ticker=ticker, strong_imbalance=threshold,
                           persistence_trades=persistence)
            engine = DiagnosticEngine(cfg, multiplier)
            engine.connected = False
            feed = Feed(engine)
            feed.event_ticker = meta['market']['event_ticker']
            engines[name][ticker] = (engine, feed)

    with gzip.open(directory / 'feed.jsonl.gz', 'rt') as source:
        for line in source:
            row = json.loads(line)
            data = row.get('data')
            if not data:
                continue
            typ = data.get('type')
            msg = data.get('msg', {})
            ticker = msg.get('market_ticker')
            raw = raw_books.get(ticker)
            if raw is not None and typ in ('orderbook_snapshot', 'orderbook_delta'):
                if typ == 'orderbook_snapshot':
                    raw.snapshot(msg, row['t'])
                else:
                    raw.delta(msg, row['t'])
            for markets in engines.values():
                for engine, feed in markets.values():
                    current_data = data
                    if (ticker == engine.cfg.ticker and typ == 'orderbook_delta'
                            and not engine.book.ready):
                        snapshot = dict(market_ticker=ticker,
                            **{side + '_dollars_fp': [
                                [format(p/10000, '.4f'), str(q)]
                                for p, q in raw.levels[side].items()]
                               for side in ('yes', 'no')})
                        current_data = dict(data, type='orderbook_snapshot', msg=snapshot)
                    feed.message(current_data, row['t'], row['wall'])

    result = {'source': str(directory), 'variants': {}}
    for name, markets in engines.items():
        per_market = {}
        for ticker, (engine, _) in markets.items():
            item = summarize(engine)
            mature_rows = [f for f in engine.forecasts
                           if f['label_status'] == 'matured' and f.get('y') in (-1, 0, 1)]
            item['causal_predictor_comparison'] = learned_stats(mature_rows)
            per_market[ticker] = item
        pooled_rows = []
        for engine, _ in markets.values():
            pooled_rows.extend(f for f in engine.forecasts
                               if f['label_status'] == 'matured' and f.get('y') in (-1, 0, 1))
        pooled = row_stats(pooled_rows)
        pooled['causal_predictor_comparison'] = learned_stats(pooled_rows)
        pooled['magnitude_bands_same_schedule'] = {
            'small_abs_lt_04': row_stats([f for f in pooled_rows
                                          if abs(f['anchor_imbalance']) < .4]),
            'medium_abs_04_to_08': row_stats([f for f in pooled_rows
                                              if .4 <= abs(f['anchor_imbalance']) < .8]),
            'large_abs_ge_08': row_stats([f for f in pooled_rows
                                          if abs(f['anchor_imbalance']) >= .8]),
        }
        for flag in ('weakened_below_08', 'crossed_hysteresis_04', 'sign_flip', 'strong_sign_flip'):
            x = [f for f in pooled_rows if f.get(flag)]
            s = sum(f['y'] == f['anchor_sign'] for f in x)
            o = sum(f['y'] == -f['anchor_sign'] for f in x)
            z = sum(f['y'] == 0 for f in x)
            pooled[flag] = dict(n=len(x), same=s, opposite=o, zero=z,
                                p_same=s/len(x) if x else None,
                                direction_accuracy_given_move=s/(s+o) if s+o else None)
        result['variants'][name] = dict(settings=dict(strong_imbalance=variants[name][0],
                                                      persistence_trades=variants[name][1],
                                                      anchor_stride=5, max_trades=50),
                                             pooled=pooled, markets=per_market)
    out = directory / 'imbalance_threshold_results.json'
    out.write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    print(json.dumps(run(args.directory)['variants'], indent=2))
