# Kalshi 30-minute active-market paper run

The complete live market-data capture ran for 30 minutes across ten markets and
was replayed causally with the locked strategy. No real orders were sent. The
portfolio started with $100 and finished at **$99.6113**.

## Portfolio result

| Leg | Funded paper trades | Gross PnL | Fees | Net PnL |
|---|---:|---:|---:|---:|
| First-passage taker | 3 | -$0.0880 | $0.3007 | **-$0.3887** |
| Maker | 0 | $0.0000 | $0.0000 | $0.0000 |
| Same-contract arbitrage | 0 | $0.0000 | $0.0000 | $0.0000 |
| **Total** | **3** | **-$0.0880** | **$0.3007** | **-$0.3887** |

Observed maximum drawdown was **$0.4828**. No funded position remained open.
The incoming trade feed lag had a 34.18 ms median and 117.28 ms p95; the replay
then applied the requested 5 ms action delay on entry and exit.

The three funded taker trades were:

| Market | Size | Rule | Exit | Gross | Fees | Net |
|---|---:|---|---|---:|---:|---:|
| Liv Hovde wins | 4 | normal | 50-trade timeout | -$0.1680 | $0.1276 | -$0.2956 |
| Wilson Leite wins | 4 | normal | 50-trade timeout | +$0.0320 | $0.1388 | -$0.1068 |
| Wilson Leite wins | 1 | reduced 45% | profitable first passage | +$0.0480 | $0.0343 | +$0.0137 |

Fees explain $0.3007, or 77% of the final loss, but they are not the entire
problem: gross PnL was already -$0.0880. The Hovde loss exceeded the two Wilson
gross gains.

## Locked stochastic first-passage signal

There were 194 fully matured, initialized forecasts. The model's most likely
state was correct on 150, or **77.3%**, but an always-ZERO prediction would have
scored **82.0%** because 159 of 194 outcomes were ZERO. Thus raw three-state
accuracy did not show edge.

The action rule generated only eight directional forecasts:

- Seven timed out as ZERO within 50 trades.
- One reached a fee-covering move, in the predicted direction.
- Three of the eight were funded; the others were blocked by an existing shared
  wallet position.
- The one correct movement gives 100% direction accuracy conditional on a move,
  but the denominator is one and is not meaningful evidence of direction skill.
- The model abstained on 186 forecasts. Of these, 152 were ZERO and 34 later had
  a fee-covering movement, so the safety filter removed many trades but also
  missed almost all movements.

The empirical probabilities were generated only from outcomes already matured
before each decision and conditioned on both imbalance bucket and observable
previous state. The weak result therefore came from sparse, unstable conditional
cells and the dominance of ZERO outcomes, rather than a fitted-model training
split or look-ahead.

## Maker results

The maker gate funded no quotes because none met the preset combination of
positive estimated net value and extreme two-sided completion safety.

The rolling initializer completed 636 overlapping hypothetical quote episodes:

| Outcome | Episodes | Share |
|---|---:|---:|
| Both sides filled | 235 | 36.9% |
| At least one side filled | 486 | 76.4% |
| One-sided fill only | 251 | 39.5% |
| No fill | 150 | 23.6% |

Their summed net was -$12.3057. This is a diagnostic rather than investable PnL
because the episodes overlap and assume independent capital. It clearly shows
why the gate refused to trade: getting the first fill was common, while completing
both sides before adverse exit was much less common.

The taker-flow distribution did contain predictive information about the next
taker side. Across 1,260 scored windows, the full state improved weighted next-side
accuracy from **65.5% to 77.1%**. For whether the opposite side arrived within
five trades, accuracy improved only from **60.3% to 62.9%**. The first statistic
does not create maker profit by itself; a paired quote needs the second fill and
must survive one-sided inventory risk.

## Arbitrage

The same-contract scanner made 396,124 valid contemporaneous book checks and
found no executable YES+NO pair below $1 after depth, slippage and both taker
fees. These checks are repeated book states, not independent opportunities.

## Data and integrity

- 418,577 complete records: 11,248 trades, 398,705 order-book deltas, 13
  snapshots and lifecycle/subscription messages.
- All ten selected markets traded during the run.
- Opening and closing fee metadata agreed.
- Three live snapshots represented fully empty books by omitting both side
  arrays. The original parser stopped rather than guessing. It was fixed to clear
  old depth for these identified snapshots, preventing phantom liquidity. It
  still rejects legacy or malformed fields.
- Fifteen combined-engine tests and 22 taker-engine tests passed after the fix.
- The capture itself was not changed; only the replay parser was corrected.

The central result is that the full stochastic first-passage strategy did not
show a tradable edge in this 30-minute sample. The maker gate correctly detected
unfavorable completion economics, while the flow model's next-side accuracy is
the only promising diagnostic and needs a longer, prespecified paper run before
it can support trading.
