from pathlib import Path

import numpy as np
import pandas as pd

from high_fidelity_replay import OUT, _probs, _update_counts, _causal_buckets
from kalshi_probability_sizing import drawdown, kalshi_fee


LOCKED_MAX_EVENTS = 50
DIAGNOSTIC_MAX_EVENTS = [10, 25, 50, 100, 250]
WARMUP_FRAC = 0.7
Q = 1

DATASETS_EXT = {
    "BTC15M": "quant_results/kalshi-btc15m-2026-09-11.txt.zst.features.csv.gz",
    "NFL_QUANT": "quant_results/kalshi-KXNFLGAME-26SEP13WASPHI-PHI-2026-09-13.txt.zst.features.csv.gz",
    "NFL_V3": "sports_v3/NFL/features.csv.gz",
    "MLB_V3": "sports_v3/MLB/features.csv.gz",
    "NFL_V4": "sports_v4/NFL/features.csv.gz",
    "MLB_V4": "sports_v4/MLB/features.csv.gz",
}


def load_market(path):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["t", "bid", "ask", "bq", "aq", "imb1"])
    df = df[(df.ask > df.bid) & (df.bid > 0) & (df.ask < 1)].copy().reset_index(drop=True)
    changed = (
        (df.bid != df.bid.shift())
        | (df.ask != df.ask.shift())
        | (df.bq != df.bq.shift())
        | (df.aq != df.aq.shift())
    )
    df = df[changed].copy().reset_index(drop=True)
    df["mid"] = (df.bid + df.ask) / 2
    df["spread"] = df.ask - df.bid
    df["abs_imb"] = df.imb1.abs()
    df["imb_bucket"] = _causal_buckets(df.imb1.to_numpy())
    return df


def yes_pnl(q, entry, exit_price):
    return q * (exit_price - entry) - kalshi_fee(q, entry) - kalshi_fee(q, exit_price)


def no_pnl(q, bid0, ask1):
    entry = 1.0 - bid0
    exit_price = 1.0 - ask1
    return q * (exit_price - entry) - kalshi_fee(q, entry) - kalshi_fee(q, exit_price)


def fp_label(df, i, max_events):
    raise RuntimeError("Retired unsafe label path. Use audited_fp.first_passage.")
    end = min(i + max_events + 1, len(df))
    if end <= i + 1:
        return 0
    bid0 = float(df.bid.iat[i])
    ask0 = float(df.ask.iat[i])
    for j in range(i + 1, end):
        yp = yes_pnl(1, ask0, float(df.bid.iat[j]))
        npnl = no_pnl(1, bid0, float(df.ask.iat[j]))
        if yp > 0 or npnl > 0:
            return 1 if yp >= npnl else -1
    return 0


def build_events(market, df, max_events):
    # Same anchor style as stock/NSE: extreme imbalance and relatively tight spread,
    # but causal tightness is simplified to an expanding median threshold.
    spread_q25 = pd.Series(df.spread).expanding(min_periods=20).quantile(0.25)
    setup = (df.abs_imb >= 0.8) & (df.spread <= spread_q25)
    rows = []
    for i in np.arange(0, len(df), 5):
        i = int(i)
        if not bool(setup.iat[i]) or i + 1 >= len(df):
            continue
        rows.append(
            {
                "market": market,
                "i": i,
                "t": float(df.t.iat[i]),
                "imb_bucket": int(df.imb_bucket.iat[i]),
                "imb1": float(df.imb1.iat[i]),
                "y": fp_label(df, i, max_events),
            }
        )
    ev = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    return ev


def signal_key(r):
    return ("imb_bucket", int(r.imb_bucket))


def update_counts(counts, r):
    _update_counts(counts, ("all",), r.y)
    _update_counts(counts, signal_key(r), r.y)


def side_from_counts(counts, r):
    pminus, pzero, pplus = _probs(counts, signal_key(r))
    return (1 if pplus >= pminus else -1), pminus, pzero, pplus


def replay(market, df, events, counts, max_events, split):
    raise RuntimeError("Retired unsafe tester. Use audited_fp_runner.py.")
    cash = 0.0
    trades = []
    next_allowed_i = -1
    for r in events.itertuples(index=False):
        if r.i < next_allowed_i:
            update_counts(counts, r)
            continue
        side, pminus, pzero, pplus = side_from_counts(counts, r)
        entry_i = int(r.i)
        bid0 = float(df.bid.iat[entry_i])
        ask0 = float(df.ask.iat[entry_i])
        end = min(entry_i + max_events + 1, len(df))
        exit_i = None
        pnl = None
        entry_price = ask0 if side == 1 else 1.0 - bid0
        exit_price = None
        for j in range(entry_i + 1, end):
            if side == 1:
                candidate_exit = float(df.bid.iat[j])
                candidate_pnl = yes_pnl(Q, entry_price, candidate_exit)
            else:
                candidate_exit = 1.0 - float(df.ask.iat[j])
                candidate_pnl = Q * (candidate_exit - entry_price) - kalshi_fee(Q, entry_price) - kalshi_fee(Q, candidate_exit)
            if candidate_pnl > 0:
                exit_i, pnl, exit_price = j, candidate_pnl, candidate_exit
                break
        if exit_i is None:
            j = end - 1
            if j > entry_i:
                if side == 1:
                    exit_price = float(df.bid.iat[j])
                    pnl = yes_pnl(Q, entry_price, exit_price)
                else:
                    exit_price = 1.0 - float(df.ask.iat[j])
                    pnl = Q * (exit_price - entry_price) - kalshi_fee(Q, entry_price) - kalshi_fee(Q, exit_price)
                exit_i = j
        if exit_i is not None:
            cash += pnl
            next_allowed_i = exit_i
            trades.append(
                {
                    "market": market,
                    "split": split,
                    "max_events": max_events,
                    "q": Q,
                    "anchor_i": entry_i,
                    "exit_i": exit_i,
                    "anchor_t": float(df.t.iat[entry_i]),
                    "exit_t": float(df.t.iat[exit_i]),
                    "wait_events": int(exit_i - entry_i),
                    "wait_s": float(df.t.iat[exit_i] - df.t.iat[entry_i]),
                    "side": side,
                    "pminus": pminus,
                    "pzero": pzero,
                    "pplus": pplus,
                    "entry": entry_price,
                    "exit": exit_price,
                    "pnl": pnl,
                    "cash": cash,
                }
            )
        update_counts(counts, r)
    t = pd.DataFrame(trades)
    return {
        "asset": market,
        "split": split,
        "signal": "rolling_imbalance_conditioned_first_passage_trade_time",
        "max_events": max_events,
        "q": Q,
        "anchors": len(events),
        "trades": len(t),
        "net": float(t.pnl.sum()) if len(t) else 0.0,
        "mean_trade": float(t.pnl.mean()) if len(t) else np.nan,
        "median_trade": float(t.pnl.median()) if len(t) else np.nan,
        "max_drawdown": drawdown(np.r_[0, t.pnl.cumsum()]) if len(t) else 0.0,
        "avg_wait_s": float(t.wait_s.mean()) if len(t) else np.nan,
        "median_wait_s": float(t.wait_s.median()) if len(t) else np.nan,
        "p90_wait_s": float(t.wait_s.quantile(0.9)) if len(t) else np.nan,
    }, t


def run_market(market, path, max_events):
    df = load_market(path)
    ev = build_events(market, df, max_events)
    if len(ev) < 20:
        return None, pd.DataFrame()
    split_i = max(5, int(len(ev) * WARMUP_FRAC))
    train = ev.iloc[:split_i].copy()
    test = ev.iloc[split_i:].copy()
    counts = {}
    for r in train.itertuples(index=False):
        update_counts(counts, r)
    return replay(market, df, test, counts, max_events, "test")


def run_all(max_events):
    rows = []
    trades = []
    for market, path in DATASETS_EXT.items():
        if not Path(path).exists():
            continue
        s, t = run_market(market, path, max_events)
        if s is not None:
            rows.append(s)
        if len(t):
            trades.append(t)
    return rows, pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()


def main():
    raise SystemExit("Legacy results invalidated. Use audited_fp_runner.py; run tests first.")
    OUT.mkdir(exist_ok=True)
    rows, trades = run_all(LOCKED_MAX_EVENTS)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "locked_fp_trade_time_kalshi_summary.csv", index=False)
    trades.to_csv(OUT / "locked_fp_trade_time_kalshi_trades.csv", index=False)
    diag_rows = []
    for n in DIAGNOSTIC_MAX_EVENTS:
        diag_rows.extend(run_all(n)[0])
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(OUT / "locked_fp_trade_time_kalshi_diagnostics.csv", index=False)
    print("KALSHI_LOCKED_TRADE_TIME")
    print(summary.to_string(index=False))
    print("\nKALSHI_DIAGNOSTICS_NOT_SELECTION")
    print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
