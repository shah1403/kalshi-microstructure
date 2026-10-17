# Independent verification

All 418,577 JSONL records were parsed from the preserved gzip capture. The offline replay's entire `combined_results.json` equals the original parsed object. Ledger gross minus fees reconciles to the combined wallet. Weighted flow metrics and quote-completion counts were independently recomputed from replay outputs.

Regression tests rerun: 15 combined/recovery tests, 22 first-passage engine tests, and 13 maker tests, all passing. They exercise snapshot reset, sequence safety, fee rounding, shared wallet behavior, latency, queue replacement, partial fills, and future-extension invariance. Passing these finite tests does not certify every behavior.

The flow model updates only matured windows and makes predictions from current/past context. Its 50-trade windows overlap; 1,260 windows are not 1,260 independent markets. Only ten markets and one 30-minute recording are represented in the central result. Model selection in earlier experiments may influence this pilot design, so it is not a pristine confirmatory holdout.

The historical three-state model had 150/194 correct matured predictions (77.3%), versus 159/194 (82.0%) for always ZERO. That metric is separate from next-taker-side accuracy. Maker completion is simulated, and there were no funded maker fills in the active run. Fee metadata was consistent at start and end; the fee implementation remains an explicit rounding/fill convention.

## Evaluation scope

- No live-order performance, brokerage statements, or independently audited investment track record.
- No universal alpha claim or annualized Sharpe from this brief, dependent sample.
- Other historical packages were recovered and indexed, not all rerun. Earlier GOOG/NSE first-passage results have documented leakage defects; see the separate first-passage project.
- Original code and figures retain historical wording. The current README and verification files control the interpretation of verified claims.

