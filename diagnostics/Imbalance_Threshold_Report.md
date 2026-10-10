# Persistent imbalance threshold diagnostic

This diagnostic uses the complete 30-minute, ten-market Kalshi capture. Every
outcome is the same executable first-passage state used by the locked taker
strategy: within 50 public trades, could one contract exit profitably after
entry and exit slippage and Kalshi fees? Direction in this diagnostic is simply
the sign of the initial top-of-book imbalance. The actual strategy remains more
general and learns either continuation or contrarian behavior from its rolling
distribution conditioned on imbalance bucket and previous stochastic state.

## What is currently implemented

Top-of-book imbalance is

\[
I=\frac{Q_{bid}-Q_{ask}}{Q_{bid}+Q_{ask}}.
\]

The current signal requires \(|I|\ge 0.8\) with the same sign at three consecutive
trade observations. At 0.8, the larger top queue is nine times the smaller top
queue. An anchor also requires the spread to be within the causal bottom quartile
of spreads observed so far and at least five trades since the previous anchor.

There is no hysteresis in the live lock counter. One trade observation below 0.8
resets it. However, an existing anchor and any funded position retain their
original direction for the full 50-trade first-passage window. Later imbalance
changes are flagged for diagnosis but do not relabel, cancel or reverse the
original forecast.

## Same-schedule magnitude comparison

These rows use the same every-five-trade observation schedule, so the comparison
isolates the magnitude of imbalance.

| Initial imbalance | Matured observations | Same-direction move | Opposite move | ZERO | Direction accuracy given a move |
|---|---:|---:|---:|---:|---:|
| Small, \(|I|<0.4\) | 466 | 3.86% | 3.86% | 92.27% | 50.0% |
| Medium, \(0.4\le|I|<0.8\) | 265 | 7.55% | 5.66% | 86.79% | 57.1% |
| Large, \(|I|\ge0.8\) | 550 | **10.18%** | 6.36% | 83.45% | **61.5%** |

Large imbalance raised the absolute chance of a fee-covering move in its own
direction from 3.86% to 10.18%, about 2.6 times the small-imbalance rate. It also
raised direction accuracy conditional on a move from chance to 61.5%. ZERO still
dominated, so magnitude alone is not an entry rule.

## Effect of persistence

| Signal rule | Matured observations | Same direction | Opposite | ZERO | Direction accuracy given a move |
|---|---:|---:|---:|---:|---:|
| Any nonzero imbalance, no persistence | 1,281 | 7.34% | 5.31% | 87.35% | 58.0% |
| Large imbalance, no persistence | 637 | 9.73% | 6.59% | 83.67% | 59.6% |
| Large imbalance for 3 trades | 502 | **9.96%** | **5.38%** | 84.66% | **64.9%** |

Persistence did not materially increase the unconditional probability of a
same-direction move beyond the large threshold alone. Its useful effect was to
reduce opposite moves and improve direction accuracy conditional on movement.

## What happened when imbalance changed

Among the 502 matured large-and-persistent anchors:

- 366, or 72.9%, fell below 0.8 strictly before the outcome.
- 340, or 67.7%, crossed a lower 0.4 hysteresis boundary strictly before the
  outcome.
- 327, or 65.1%, flipped sign at least briefly.
- 305, or 60.8%, reached a large imbalance in the opposite direction.

All 50 same-direction moves occurred after the imbalance had already fallen
below 0.8. Forty-eight occurred after it crossed 0.4, 43 after at least a brief
sign flip, and 30 after a large opposite imbalance. The exact book update that
created first passage is excluded from those change counts.

This is consistent with queue depletion: the initial imbalance can contain
information, but the book must change before price moves. Therefore, canceling
the prediction immediately when the imbalance weakens would destroy the observed
signal. A sticky state makes more sense than a hard cancellation rule.

If hysteresis is added, the cleaner implementation is to lock at 0.8 after three
trades and preserve the original state through ordinary weakening. A new opposite
state should require its own persistent evidence. The present experiment does
not establish the optimal unlock or reversal rule; testing that requires pricing
the exit at the exact time each proposed reversal rule fires.

The observations overlap and come from one 30-minute session, so they are not
independent evidence of a deployable edge. They do show a monotonic threshold
effect and support keeping the initial imbalance state alive rather than treating
every small book fluctuation as a new signal.

## Causal non-assumed distribution test

The next test removed the assumption that positive imbalance predicts UP and
negative imbalance predicts DOWN. At every initialized anchor, it compared:

1. The raw sign of current imbalance.
2. The rolling distribution of all prior outcomes in that market.
3. The rolling distribution conditioned on imbalance bucket.
4. The rolling distribution conditioned on imbalance bucket and the observable
   previous stochastic state.

Every distribution used only labels that had completely matured before the
forecast. Direction was selected by comparing its prior probabilities of DOWN
and UP. A tie produced no directional prediction. The comparison below uses the
same 224 initialized large-and-persistent observations; 35 eventually produced
a fee-covering movement.

| Direction method | Directional predictions | Correct | Direction accuracy |
|---|---:|---:|---:|
| Raw imbalance sign | 35 | 26 | **74.3%** |
| Whole-history distribution | 25 | 6 | 24.0% |
| Imbalance-conditioned distribution | 25 | 9 | 36.0% |
| Imbalance + previous-state distribution | 18 | 10 | **55.6%** |

The full conditioned distribution improved substantially over the other rolling
distributions, but it did not beat the raw imbalance sign. It tied on 17 of the
35 movements because its local cells had equal UP and DOWN counts. On its 18
non-tied forecasts, it agreed with the current imbalance only 16.7% of the time,
so it had mostly learned a contrarian mapping.

That contrarian mapping was unstable across markets. It helped Pol Martin Tiffon,
where raw imbalance direction was correct on 2 of 7 movements, but hurt Liv
Hovde, where raw direction was correct on 12 of 16 while the full conditional
distribution was correct on only its 2 of 6 non-tied predictions. Guinea-Bissau
had 10 of 10 moves in the raw imbalance direction, while the conditional cells
remained tied. Wilson Leite had only two initialized movements and both methods
were correct.

The deployed action threshold was still more selective: eight directional
signals, seven ZERO outcomes and one actual movement, correctly directed. Only
three could be funded under the shared-wallet replay, producing the previously
reported -$0.3887 net PnL.

The likely problem is not that conditioning is inherently wrong. The 30-minute
sample produces sparse, overlapping conditional cells while the relationship
between imbalance and direction changes by market and over time. The rolling
distribution reacts slowly to that change and can preserve an earlier contrarian
regime after the live relationship has become continuation. In this sample, the
current large persistent imbalance was more informative than that lagged history.
