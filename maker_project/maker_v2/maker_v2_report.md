# Maker v2 and paired-taker arbitrage: completed replay

**The raw data shows predictive information about order-flow timing. It does not establish robust maker profitability.** The strict 45–55% maker placed no trades; the wider-band MLB dynamic diagnostic made $0.0832 after fees from five filled episodes. One $0.1819 gain accounts for more than the entire net profit.

## What was added

The model now forecasts separate buy and sell waiting distributions, the next opposite-side take, a joint next-side/size-bin distribution, side-specific size distributions, and cumulative buy/sell volume at 1/2/5/10/20/50 trades. It learns empirical queue-fill outcome/time distributions from completed counterfactual quote episodes. The maker can cancel/replace quotes as predictions change, loses queue priority when repricing, reassesses the remaining inventory, and exits when the policy detects adverse flow or the odds leave the permitted band.

These are expanding empirical distributions. No regression or parameter fit replaces the counts. Prices, features and updates are causal, and extending a dataset into the future was tested not to change earlier forecasts. A label enters the distribution only after its entire observation window has matured, with strictly earlier receipt time than the current decision. Flow initialization is 100 completed windows; maker initialization is 30 completed quote episodes per quote placement. The context is known at decision time; later imbalance changes enter later decisions.

## Forward accuracy

All-odds numbers below score valid continuous recording windows. They are **not restricted to 45–55%**, and they are not executable-trade directional accuracy. Forecasts occur every fifth actual trade. The target next side means the very next subsequent trade.

| Sample | Predictions | Unconditional side accuracy | Repeat-last-side | Full side accuracy | Opposite ≤5: baseline → full |
|---|---:|---:|---:|---:|---:|
| MLB Texas–Seattle | 3,324 | 68.0% | 77.8% | 75.8% | 54.0% → 65.8% |
| NFL Washington–Philadelphia | 8,683 | 84.8% | 84.8% | 86.5% | 61.3% → 72.3% |
| BUF–JAC / BUF (sparse) | 1,247 | 71.9% | 72.3% | 72.2% | 64.1% → 71.4% |
| BUF–JAC / JAC (sparse) | 415 | 64.8% | 74.7% | 69.4% | 70.8% → 67.5% |

The full model beats unconditional frequencies on raw MLB and NFL, but **loses to simple side persistence on MLB**. This prevents the 75.8% headline from being presented as a superior new predictor. In the actual 45–55% MLB regime, full and unconditional next-side accuracy were both 90.7% on 535 forecasts. Opposite-within-five accuracy improved from 69.0% to 74.8% there. There are no 45–55% NFL observations. Sparse BUF–JAC figures are diagnostics from selected intact short stretches, not independent clean validation.

| Raw sample | Mean absolute error in capped opposite wait, baseline → full | Next size-bin accuracy, baseline → full | Next observed midpoint direction, baseline → full |
|---|---:|---:|---:|
| MLB Texas–Seattle | 12.17 → 9.27 trades | 44.0% → 42.3% | 48.5% → 59.4% |
| NFL Washington–Philadelphia | 16.77 → 11.96 trades | 42.8% → 44.7% | 47.7% → 63.4% |

“Opposite ≤5” asks whether at least one opposite-aggressor trade appears in the next five trades. Accuracy uses a 50% decision threshold. Timing error is the error in the predicted restricted mean wait, capped at 50; no observed arrival before the episode stop receives the 50-trade penalty. It is not a promise of eventual execution. Size bins are <1, [1,10), [10,100), [100,1000), and ≥100 contracts. Midpoint direction excludes no-change outcomes and means the next change visible at a trade observation, not necessarily every intratrade book flicker.

Full JSON results additionally contain per-side recall, class-balanced accuracy, probability squared errors, waiting-time CDFs in trades and seconds, side-specific expected volume errors, conditional expected waits, disjoint-window checks and five-minute-block bootstrap intervals. Brier scores are mean squared probability errors: lower is better. Accuracy alone can be high by repeatedly choosing the more common event.

The 50-trade windows overlap. In disjoint raw windows, next-side accuracy was 78.4% versus 68.2% on MLB (305 forecasts), and 86.1% versus 82.9% on NFL (791). These remain two game recordings, not evidence of cross-market robustness.

## Predicting our queue fills is harder

| Sample / odds band | Scored quotes | Any-fill accuracy, baseline → full | Full model: predicted fill rate / observed |
|---|---:|---:|---:|
| MLB Texas–Seattle / 45–55% | 26 | 3.8% → 26.9% | 48.9% / 53.8% |
| MLB Texas–Seattle / 30–70% | 131 | 67.2% → 74.0% | 63.2% / 81.7% |
| NFL Washington–Philadelphia / 30–70% | 136 | 66.2% → 64.0% | 64.4% / 66.2% |

“Any fill” includes a partial fill. The main 45–55% MLB quote sample was small and poorly predicted. In the wider diagnostic, extra conditioning helped MLB any-fill accuracy but hurt NFL. Average fill-rate calibration is not the same as identifying which individual quotes fill. Both-full-pair accuracy of 81.7% MLB / 91.2% NFL in the wide band merely matched always predicting “not both”; it is not an edge.

Quote labels are hypothetical fills under the queue model, not actual private order fills. The learner uses all available matured eligible quote episodes from the original 30–70% and 45–55% shadow runs, deduplicated by timestamp and placement. This expands the primary regime's initialization information, which is why its scored-quote count differs from v1. Flow learning uses all valid odds. Neither dataset eligibility nor features are derived from future ranks/quantiles.

## PnL, $5 initial bankroll per contract

| Sample | Odds band | Spread-only net | Fixed probability net | Dynamic gross | Dynamic fees | Dynamic net | Dynamic DD | Filled / both paired |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MLB Texas–Seattle | 0.45_0.55 | $-0.3112 | $+0.0000 | $+0.0000 | $0.0000 | $+0.0000 | $0.0000 | 0 / 0 |
| MLB Texas–Seattle | 0.30_0.70 | $-0.3777 | $-0.0592 | $+0.1570 | $0.0738 | $+0.0832 | $0.1199 | 5 / 2 |
| NFL Washington–Philadelphia | 0.45_0.55 | $+0.0000 | $+0.0000 | $+0.0000 | $0.0000 | $+0.0000 | $0.0000 | 0 / 0 |
| NFL Washington–Philadelphia | 0.30_0.70 | $-0.4775 | $+0.0000 | $+0.0000 | $0.0000 | $+0.0000 | $0.0000 | 0 / 0 |
| BUF–JAC / BUF (sparse) | 0.45_0.55 | $-1.7222 | $+0.0000 | $+0.0000 | $0.0000 | $+0.0000 | $0.0000 | 0 / 0 |
| BUF–JAC / BUF (sparse) | 0.30_0.70 | $-3.3231 | $+0.0197 | $+0.0080 | $0.0476 | $-0.0396 | $0.0517 | 3 / 1 |
| BUF–JAC / JAC (sparse) | 0.45_0.55 | $-1.8230 | $+0.0000 | $+0.0000 | $0.0000 | $+0.0000 | $0.0000 | 0 / 0 |
| BUF–JAC / JAC (sparse) | 0.30_0.70 | $-3.4343 | $+0.0000 | $+0.0000 | $0.0000 | $+0.0000 | $0.0000 | 0 / 0 |

The primary band remains 45–55%; 30–70% is a separately declared diagnostic. One contract per side is used to isolate the signal. Quotes have a maximum 50-trade lifetime and a 300-second stale-order guard. A dynamic policy can cancel earlier for negative updated expected edge, low opposite-flow probability, adverse midpoint probability, or leaving the odds band.

The MLB dynamic diagnostic has two fully paired wins and three one-sided losing exits. The largest win was buying YES at $0.52 and NO at $0.29, giving $0.19 gross / $0.1819 net. Excluding that one episode leaves −$0.0987. The single 20 ms execution sensitivity produced $+0.1459 from 4 filled episodes; this is not a replacement for the requested 5 ms baseline or a latency optimum. It changes which cancellations/fills occur and the sample is much too small for a robustness claim.

The v2 baseline differs slightly from v1 because candidates are now checked every fifth trade globally, accounting supports changing quote prices, and fee-plus-position-cost rounding handles fractional fills. No previous result is silently substituted. The fee correction changes the wide MLB static result by five thousandths of a cent.

Historical fixed-quote oracles, supplied as opportunity context (their candidate grid/management differs from v2's dynamic policy):
- MLB Texas–Seattle, primary_45_55: $+0.0524.
- MLB Texas–Seattle, diagnostic_30_70: $+1.0112.
- NFL Washington–Philadelphia, primary_45_55: $+0.0000.
- NFL Washington–Philadelphia, diagnostic_30_70: $+0.1058.
- BUF–JAC / BUF (sparse), primary_45_55: $+0.4874.
- BUF–JAC / BUF (sparse), diagnostic_30_70: $+1.2927.
- BUF–JAC / JAC (sparse), primary_45_55: $+0.2068.
- BUF–JAC / JAC (sparse), diagnostic_30_70: $+0.6297.

These are greedy hindsight selectors, not globally optimal PnL bounds. It would be invalid to divide v2 profit by them and call that a precise fraction of available alpha captured.

## Paired YES + NO taker arbitrage

For equal quantity q of YES and NO on the **same complementary binary contract**, the normal aggregate settlement payoff is q dollars. A locked spread requires both actual fills and `YES acquisition cost + NO acquisition cost + both taker fees < q`. Displayed last prices, bids, mismatched contract definitions, or a single filled leg do not establish that trade.

In a consistent Kalshi YES book, `YES ask + NO ask = YES ask + (1 - YES bid) = 1 + spread`. Thus a same-contract pair below $1 would require crossed offers in this representation. Crossed or stale reconstructed books are quarantined. The audit found **zero net-profitable one-contract paired-taker candidates after the modeled fees/slippage in all four feeds**, even before granting execution to any crossed snapshot. No paired arbitrage was executed; certified-pair count, locked settlement value and arbitrage PnL are all zero. This does not prove such opportunities never occur elsewhere.

The separate paired-taker state machine was unit-tested with synthetic prices: first and second legs each have latency, limits and depth checks; a failed second leg leads to a delayed unwind; a completed pair locks its capital until settlement. Its synthetic profit is never included in historical results. Historical settlement records are unnecessary for this run because there are no held pairs. The maker cash ledger offsets opposite exposure as round-trip trading; it is separate from the hold-to-settlement arbitrage ledger.

## Fidelity, controls and remaining limits

- **5 ms additional latency per entry/cancel action**, on top of the vendor's receipt lag. Repricing waits for cancel acknowledgement and a further entry delay. Forced exits wait another 5 ms after cancellation. The 50-trade trigger initiates cancellation; fills can still arrive during its latency, so realized fills can occasionally exceed trade 50.
- FIFO assumption: all displayed same-price quantity starts ahead; only matching signed trade volume depletes it; cancellation volume receives no priority credit. Price-touch alone cannot fill. Through-trades can fill at our resting price, subject to observed trade quantity. Replacement loses priority. Partial fills carry their actual cost basis across repricing.
- Post-only cross checks, cash reservation, one active quote episode, maker/taker fees on each fill, 50% displayed depth available to forced liquidation, and 0.1¢ extra adverse slippage per taker contract. No missing-depth fake closes; missing data censors learning and flags exposed trades.
- July 7, 2026 fee formula: maker 0.0175 and taker 0.07 times q p(1−p), multiplier 1, fee plus position cost rounded to $0.0001. January data uses this **current-fee scenario**, not certified historical January charges. Illustrative whole-cent fee tables remain inconsistent with the formula; actual live fees must be reconciled before deployment.
- Every closed ledger was independently reconciled from fill cashflows, paired offsets and fees: **686 passed**. Thirteen deterministic tests passed, including future-prefix invariance, censor handling, partial-fill basis, cancellation races, queue resets, fractional rounding, delayed exits and two-leg arbitrage failure.
- Models condition on odds and recent state, imbalance, separate taker count/volume shares and arrival rates, ages since each side's last trade, the last three sides, spread and depth/volume ratios. Full contexts back off through odds+last-side and imbalance+last-midpoint-state cells to all history with strength 20. Therefore the “imb” ablation includes midpoint state as well as imbalance; it does not isolate imbalance alone.
- Quote decisions use a probability mixture of none / YES-only / NO-only / both / partial-pair outcomes with conditional net returns. The planned paired spread is substituted for that outcome at the current quote. This remains dependent on estimated one-sided exit losses, not merely a high probability that some trade arrives. Dynamic post-fill cancellation uses the needed taker-side arrival CDF and midpoint risk; it is not an exact exchange-queue hazard model.
- Sparse BUF–JAC has large gaps; its two contracts are the same game. Reported sparse losses and drawdowns cannot bound unobserved fills during gaps. GOOG lacks a trade tape and NSE samples lack precise trade timestamps for this same latency/queue test; they are excluded rather than given invented maker fills.
- Market-by-price data cannot reveal exact cancellations ahead, hidden orders, own impact or matching-engine order sequence. Only live own-order records can validate actual fill probabilities. No claim is made that this is an error-free exchange emulator, that the 50/50 odds band is intrinsically safe, or that the model is ready for unattended real-money trading.

## Reproduction

Install root requirements. Keep the original raw files under `inputs/` and the included v1 shadow labels under `results/`. Run `python -m unittest discover -s maker_v2 -p 'test_v2.py'`, `python maker_v2/run.py`, `python maker_v2/latency_check.py`, then `python maker_v2/report.py`. `run.py` also accepts dataset names to resume selected assets. The bundle contains source, compact frozen side/size/midpoint/opposite-wait forecasts, ledgers, scores, calibration, input hashes and the legacy labels needed for reproduction; original input bytes are already in the earlier datasets and are not duplicated.

Official references: [Kalshi order-book representation](https://docs.kalshi.com/getting_started/orderbook_responses); [fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf). These establish market conventions and fee assumptions, not the validity of hypothetical fills.
