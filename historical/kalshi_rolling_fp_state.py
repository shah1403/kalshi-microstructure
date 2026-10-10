from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_probability_sizing import DATASETS, drawdown, kalshi_fee


OUT = Path("kalshi_rolling_fp_state")
OUT.mkdir(exist_ok=True)

HORIZONS_S = [1, 5, 10, 30, 60]


def load_market(path):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["t", "bid", "ask", "bq", "aq", "imb1"])
    df = df[(df.ask > df.bid) & (df.bid > 0) & (df.ask < 1)].copy()
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
    df["imb_sign"] = np.sign(df.imb1).astype(int)
    df["spread_rank"] = df.spread.rank(pct=True)
    add_stability(df)
    return df.reset_index(drop=True)


def add_stability(df):
    x = df.imb1.to_numpy()
    edges = day_edges(x)
    b = np.searchsorted(edges, x, side="right") - 1
    s = np.sign(x).astype(int)
    for lb in [3, 5, 10]:
        stable_bucket = np.zeros(len(df), dtype=int)
        stable_sign = np.zeros(len(df), dtype=int)
        for i in range(len(df)):
            lo = max(0, i - lb + 1)
            stable_bucket[i] = int(np.all(b[lo : i + 1] == b[i]))
            stable_sign[i] = int(np.all(s[lo : i + 1] == s[i]))
        df[f"stable_bucket_{lb}"] = stable_bucket
        df[f"stable_sign_{lb}"] = stable_sign


def day_edges(x, bins=5):
    vals = x[np.isfinite(x)]
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def bucket(edges, value):
    return int(np.searchsorted(edges, value, side="right") - 1)


def yes_pnl_one(bid0, ask0, bid1, ask1):
    entry = ask0
    exit_price = bid1
    return (exit_price - entry) - kalshi_fee(1, entry) - kalshi_fee(1, exit_price)


def no_pnl_one(bid0, ask0, bid1, ask1):
    entry = 1.0 - bid0
    exit_price = 1.0 - ask1
    return (exit_price - entry) - kalshi_fee(1, entry) - kalshi_fee(1, exit_price)


def first_passage_label(df, i, horizon_s):
    t = df.t.to_numpy()
    end = np.searchsorted(t, t[i] + horizon_s, side="right")
    if end <= i + 1:
        return None
    bid0 = float(df.bid.iat[i])
    ask0 = float(df.ask.iat[i])
    yes_hit = None
    no_hit = None
    yes_hit_pnl = None
    no_hit_pnl = None
    for j in range(i + 1, end):
        yp = yes_pnl_one(bid0, ask0, float(df.bid.iat[j]), float(df.ask.iat[j]))
        npnl = no_pnl_one(bid0, ask0, float(df.bid.iat[j]), float(df.ask.iat[j]))
        if yes_hit is None and yp > 0:
            yes_hit = j
            yes_hit_pnl = yp
        if no_hit is None and npnl > 0:
            no_hit = j
            no_hit_pnl = npnl
        if yes_hit is not None and no_hit is not None:
            break
    fixed_j = end - 1
    fixed_yes = yes_pnl_one(bid0, ask0, float(df.bid.iat[fixed_j]), float(df.ask.iat[fixed_j]))
    fixed_no = no_pnl_one(bid0, ask0, float(df.bid.iat[fixed_j]), float(df.ask.iat[fixed_j]))
    if yes_hit is None and no_hit is None:
        y, hit, hit_pnl = 0, np.nan, 0.0
    elif no_hit is None or (yes_hit is not None and yes_hit <= no_hit):
        y, hit, hit_pnl = 1, yes_hit, yes_hit_pnl
    else:
        y, hit, hit_pnl = -1, no_hit, no_hit_pnl
    fixed_y = 0
    if fixed_yes > 0 or fixed_no > 0:
        fixed_y = 1 if fixed_yes >= fixed_no else -1
    return {
        "y": y,
        "hit": hit,
        "hit_pnl": hit_pnl,
        "fixed_y": fixed_y,
        "fixed_j": fixed_j,
        "fixed_yes_pnl": fixed_yes,
        "fixed_no_pnl": fixed_no,
    }


def build_events(market, df):
    rows = []
    edges = day_edges(df.imb1.to_numpy())
    setups = {
        "extreme_tight": (df.abs_imb >= 0.8) & (df.spread_rank <= 0.25),
        "stable5": (df.abs_imb >= 0.8) & (df.spread_rank <= 0.25) & (df.stable_bucket_5 == 1),
        "stable10": (df.abs_imb >= 0.8) & (df.spread_rank <= 0.25) & (df.stable_bucket_10 == 1),
    }
    eligible = np.arange(0, len(df), 5)
    for setup, mask in setups.items():
        idxs = [int(i) for i in eligible if bool(mask.iat[int(i)])]
        for horizon in HORIZONS_S:
            for i in idxs:
                lab = first_passage_label(df, i, horizon)
                if lab is None:
                    continue
                rows.append(
                    {
                        "market": market,
                        "anchor": i,
                        "setup": setup,
                        "horizon_s": horizon,
                        "t": float(df.t.iat[i]),
                        "imb1": float(df.imb1.iat[i]),
                        "imb_sign": int(np.sign(df.imb1.iat[i])),
                        "imb_bucket": bucket(edges, float(df.imb1.iat[i])),
                        "bid0": float(df.bid.iat[i]),
                        "ask0": float(df.ask.iat[i]),
                        "exit_bid": float(df.bid.iat[lab["fixed_j"]]),
                        "exit_ask": float(df.ask.iat[lab["fixed_j"]]),
                        **lab,
                    }
                )
    return pd.DataFrame(rows).sort_values(["market", "t", "setup", "horizon_s"]).reset_index(drop=True)


def key_for_mode(r, mode):
    if mode == "base":
        return ("all",)
    if mode == "sign":
        return (int(r.imb_sign),)
    if mode == "bucket":
        return (int(r.imb_bucket),)
    if mode == "sign_prev":
        return (int(r.imb_sign), int(r.prev_y))
    if mode == "bucket_prev":
        return (int(r.imb_bucket), int(r.prev_y))
    raise ValueError(mode)


def update_counts(counts, key, y):
    if key not in counts:
        counts[key] = np.zeros(3)
    counts[key][{-1: 0, 0: 1, 1: 2}[int(y)]] += 1


def probs(counts, key):
    c = counts.get(key)
    if c is None or c.sum() < 5:
        c = counts.get(("all",), np.zeros(3))
    return (c + 0.5) / (c.sum() + 1.5)


def pnl_side(r, side):
    if r.y == side and np.isfinite(r.hit):
        return float(r.hit_pnl)
    if side == 1:
        return yes_pnl_one(r.bid0, r.ask0, r.exit_bid, r.exit_ask)
    return no_pnl_one(r.bid0, r.ask0, r.exit_bid, r.exit_ask)


def fixed_pnl_side(r, side):
    return float(r.fixed_yes_pnl if side == 1 else r.fixed_no_pnl)


def rolling_eval(events, warmup=30):
    rows = []
    trades = []
    modes = ["base", "sign", "bucket", "sign_prev", "bucket_prev"]
    for (market, setup, horizon), ev0 in events.groupby(["market", "setup", "horizon_s"], sort=False):
        ev = ev0.sort_values("t").reset_index(drop=True).copy()
        ev["prev_y"] = ev.y.shift(1).fillna(0).astype(int)
        if len(ev) <= warmup:
            continue
        for mode in modes:
            counts = {}
            for r in ev.iloc[:warmup].itertuples(index=False):
                update_counts(counts, ("all",), r.y)
                update_counts(counts, key_for_mode(r, mode), r.y)
            spnls, cpnls, sfpnls, cfpnls = [], [], [], []
            agree, corr, ccorr, fpagree = [], [], [], []
            for r in ev.iloc[warmup:].itertuples(index=False):
                pminus, pzero, pplus = probs(counts, key_for_mode(r, mode))
                side = 1 if pplus >= pminus else -1
                contra_side = -int(r.imb_sign) if int(r.imb_sign) != 0 else side
                sp, cp = pnl_side(r, side), pnl_side(r, contra_side)
                sfp, cfp = fixed_pnl_side(r, side), fixed_pnl_side(r, contra_side)
                spnls.append(sp)
                cpnls.append(cp)
                sfpnls.append(sfp)
                cfpnls.append(cfp)
                agree.append(int(side == contra_side))
                corr.append(int(r.y == side))
                ccorr.append(int(r.y == contra_side))
                if r.fixed_y != 0 and r.y != 0:
                    fpagree.append(int(r.fixed_y == r.y))
                trades.append(
                    {
                        "market": market,
                        "setup": setup,
                        "horizon_s": horizon,
                        "mode": mode,
                        "t": r.t,
                        "imb1": r.imb1,
                        "y": r.y,
                        "fixed_y": r.fixed_y,
                        "pminus": pminus,
                        "pzero": pzero,
                        "pplus": pplus,
                        "stoch_side": side,
                        "contra_side": contra_side,
                        "stoch_pnl": sp,
                        "contra_pnl": cp,
                        "stoch_fixed_pnl": sfp,
                        "contra_fixed_pnl": cfp,
                    }
                )
                update_counts(counts, ("all",), r.y)
                update_counts(counts, key_for_mode(r, mode), r.y)
            rows.append(
                {
                    "market": market,
                    "setup": setup,
                    "horizon_s": horizon,
                    "mode": mode,
                    "events": len(ev),
                    "warmup": warmup,
                    "trades": len(spnls),
                    "stoch_pnl": float(np.sum(spnls)),
                    "contra_pnl_same_anchors": float(np.sum(cpnls)),
                    "stoch_fixed_pnl_same_anchors": float(np.sum(sfpnls)),
                    "contra_fixed_pnl_same_anchors": float(np.sum(cfpnls)),
                    "stoch_maxdd": drawdown(np.r_[0, np.cumsum(spnls)]),
                    "contra_maxdd": drawdown(np.r_[0, np.cumsum(cpnls)]),
                    "stoch_accuracy": float(np.mean(corr)),
                    "contra_accuracy": float(np.mean(ccorr)),
                    "stoch_contra_side_agreement": float(np.mean(agree)),
                    "fixed_fp_side_agreement": float(np.mean(fpagree)) if fpagree else np.nan,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(trades)


def main():
    all_events = []
    audits = []
    for market, path in DATASETS.items():
        df = load_market(path)
        ev = build_events(market, df)
        all_events.append(ev)
        audits.append({"market": market, "book_rows": len(df), "events": len(ev)})
    events = pd.concat(all_events, ignore_index=True)
    summary, trades = rolling_eval(events)
    summary = summary.sort_values(["market", "setup", "horizon_s", "stoch_pnl"], ascending=[True, True, True, False])
    pd.DataFrame(audits).to_csv(OUT / "audit.csv", index=False)
    events.to_csv(OUT / "first_passage_anchor_events.csv", index=False)
    summary.to_csv(OUT / "rolling_fp_state_summary.csv", index=False)
    trades.to_csv(OUT / "rolling_fp_state_trades.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
