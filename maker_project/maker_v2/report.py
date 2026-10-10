from pathlib import Path
import json,gzip,zipfile,hashlib,os,tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;R=HERE/'results'
NAMES={'KXMLBGAME_MLB':'MLB Texas–Seattle','KXNFLGAME_NFL':'NFL Washington–Philadelphia','BUFJAC_BUF':'BUF–JAC / BUF (sparse)','BUFJAC_JAC':'BUF–JAC / JAC (sparse)'}
D={name:json.loads((R/(name+'_accuracy.json')).read_text()) for name in NAMES}
pnl=pd.read_csv(R/'pnl.csv')
def pc(v):return f'{100*v:.1f}%'
def money(v):return f'${v:+.4f}'

# Reconcile simulator accounting from the independent fill-level ledger.
checked=0
for path in R.glob('*.json'):
    data=json.loads(path.read_text())
    if 'ledger' not in data:continue
    for ep in data['ledger']['episodes']:
        if ep['net'] is None:continue
        costs=sum(f['qty']*f['price'] for f in ep['fills'] if f['maker'])
        proceeds=sum(f['qty']*f['price'] for f in ep['fills'] if not f['maker'])
        fees=sum(f['fee'] for f in ep['fills']);gross=ep['matched']+proceeds-costs
        assert abs(ep['gross']-gross)<1e-7 and abs(ep['net']-gross+fees)<1e-7
        assert max(ep['filled'])<=1+1e-8
        checked+=1
(R/'accounting_validation.json').write_text(json.dumps(dict(closed_episode_ledgers_checked=checked,passed=True)))

accuracy=[];cal=[]
for name,d in D.items():
    fs=d['flow_accuracy']
    for level in ['all','coarse','imb','full']:
        s=fs[level]
        accuracy.append(dict(dataset=name,scope='all_valid_odds',model=level,n=s['n'],
            side_accuracy=s['next_taker_accuracy'],opposite_within_5_accuracy=s['opposite_by_h_accuracy'][2],
            opposite_within_5_brier=s['opposite_by_h_brier'][2],capped_wait_mae=s['capped_wait_mae'],
            midpoint_direction_accuracy=s['midpoint_direction_accuracy'],size_bin_accuracy=s['next_size_bin_accuracy']))
    # Calibration includes all issued, subsequently scorable prequential predictions.
    with gzip.open(R/(name+'_flow_predictions.jsonl.gz'),'rt') as f:rows=[json.loads(x) for x in f]
    with gzip.open(R/(name+'_compact_forecasts.jsonl.gz'),'wt') as out:
        for r in rows:
            compact={k:r[k] for k in ['i','t','mid','last_side','n','actual_side','actual_size','actual_mid','truth']}
            compact['forecast']={level:{k:f[k] for k in ['side','size','mid','opp']} for level,f in r['forecast'].items()}
            out.write(json.dumps(compact)+'\n')
    for endpoint in ['next_buy','opposite_by5']:
        ps=np.array([r['forecast']['full']['side'][0] if endpoint=='next_buy' else r['forecast']['full']['opp']['cdf_trades'][2] for r in rows])
        ys=np.array([r['actual_side']==0 if endpoint=='next_buy' else r['truth']['opp_trades'] is not None and r['truth']['opp_trades']<=5 for r in rows])
        for k in range(10):
            mask=(ps>=k/10)&(ps<(k+1)/10 if k<9 else ps<=1)
            if mask.any():cal.append(dict(dataset=name,endpoint=endpoint,n=int(mask.sum()),forecast=float(ps[mask].mean()),observed=float(ys[mask].mean())))
pd.DataFrame(accuracy).to_csv(HERE/'maker_v2_accuracy.csv',index=False)
pd.DataFrame(cal).to_csv(R/'calibration.csv',index=False)

fig,axes=plt.subplots(2,2,figsize=(13,9));colors=['#79899e','#df9e3b','#167e80']
raw=list(NAMES)[:2]
for ax,endpoint,title in [(axes[0,0],'side','Next taker side'),(axes[0,1],'opposite','Opposite take within five trades')]:
    x=np.arange(2);width=.25
    for j,label in enumerate(['Rolling unconditional','Repeat last side','Full conditioned'] if endpoint=='side' else ['Rolling unconditional','Odds + last side','Full conditioned']):
        vals=[]
        for name in raw:
            s=D[name]['flow_accuracy']
            if endpoint=='side':v=s['all']['next_taker_accuracy'] if j==0 else s['persistence_accuracy'] if j==1 else s['full']['next_taker_accuracy']
            else:v=s[['all','coarse','full'][j]]['opposite_by_h_accuracy'][2]
            vals.append(v*100)
        bars=ax.bar(x+(j-1)*width,vals,width,label=label,color=colors[j]);ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
    ax.set_xticks(x,['MLB','NFL']);ax.set_ylim(0,100);ax.set_ylabel('Correct predictions (%)');ax.set_title(title);ax.legend(fontsize=8,loc='lower left')
ax=axes[1,0];ax.plot([0,1],[0,1],ls='--',c='#999',label='Perfect calibration')
for name,c in zip(raw,['#167e80','#a35e91']):
    rr=[r for r in cal if r['dataset']==name and r['endpoint']=='opposite_by5']
    ax.plot([r['forecast'] for r in rr],[r['observed'] for r in rr],'o-',c=c,label='MLB' if 'MLB' in name else 'NFL')
ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('Predicted probability');ax.set_ylabel('Observed fraction');ax.set_title('Opposite-within-five probability calibration');ax.legend(fontsize=8)
ax=axes[1,1]
for name,c,label in [('spread_only_v2','#b14c56','Spread-only'),('probability_static','#df9e3b','Probability, fixed quotes'),('probability_dynamic','#167e80','Probability, dynamic quotes')]:
    d=json.loads((R/('KXMLBGAME_MLB_0.30_0.70_'+name+'.json')).read_text());curve=d['ledger']['equity']
    if curve:
        curve=np.array(curve);ax.step(pd.to_datetime(curve[:,0],unit='ns',utc=True),curve[:,1]-5,where='post',label=label,c=c)
ax.axhline(0,lw=.6,c='#999');ax.set_title('MLB PnL: 30–70% diagnostic, $5 initial cash');ax.set_ylabel('Net paper PnL ($)');ax.legend(fontsize=8)
ax.tick_params(axis='x',rotation=20)
for ax in axes.flat:ax.grid(axis='y',alpha=.15)
fig.suptitle('Forward predictions show timing information; maker profitability remains unproven',fontsize=14)
fig.tight_layout();fig.savefig(HERE/'maker_v2_results.png',dpi=165);plt.close(fig)

flowtable=['| Sample | Predictions | Unconditional side accuracy | Repeat-last-side | Full side accuracy | Opposite ≤5: baseline → full |',
           '|---|---:|---:|---:|---:|---:|']
for name,d in D.items():
    s=d['flow_accuracy'];a=s['all'];f=s['full']
    flowtable.append(f"| {NAMES[name]} | {f['n']:,} | {pc(a['next_taker_accuracy'])} | {pc(s['persistence_accuracy'])} | {pc(f['next_taker_accuracy'])} | {pc(a['opposite_by_h_accuracy'][2])} → {pc(f['opposite_by_h_accuracy'][2])} |")
timing=['| Raw sample | Mean absolute error in capped opposite wait, baseline → full | Next size-bin accuracy, baseline → full | Next observed midpoint direction, baseline → full |', '|---|---:|---:|---:|']
for name in raw:
    s=D[name]['flow_accuracy'];a=s['all'];f=s['full'];timing.append(f"| {NAMES[name]} | {a['capped_wait_mae']:.2f} → {f['capped_wait_mae']:.2f} trades | {pc(a['next_size_bin_accuracy'])} → {pc(f['next_size_bin_accuracy'])} | {pc(a['midpoint_direction_accuracy'])} → {pc(f['midpoint_direction_accuracy'])} |")
fills=['| Sample / odds band | Scored quotes | Any-fill accuracy, baseline → full | Full model: predicted fill rate / observed |', '|---|---:|---:|---:|']
for name in raw:
    for band in ['45_55','30_70']:
        q=D[name]['quote_accuracy'][band];a=q['all'];f=q['full']
        if f['n']:fills.append(f"| {NAMES[name]} / {band.replace('_','–')}% | {f['n']} | {pc(a['any_fill_accuracy'])} → {pc(f['any_fill_accuracy'])} | {pc(f['forecast_any_fill'])} / {pc(f['actual_any_fill'])} |")
pnlt=['| Sample | Odds band | Spread-only net | Fixed probability net | Dynamic gross | Dynamic fees | Dynamic net | Dynamic DD | Filled / both paired |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for name in NAMES:
    for band in ['0.45_0.55','0.30_0.70']:
        b=pnl[(pnl.dataset==name)&(pnl.band==band)].set_index('variant');s=b.loc['probability_dynamic']
        pnlt.append(f"| {NAMES[name]} | {band} | {money(b.loc['spread_only_v2','net'])} | {money(b.loc['probability_static','net'])} | {money(s.gross)} | ${s.fees:.4f} | {money(s.net)} | ${s.maxdd:.4f} | {int(s.filled)} / {int(s.paired)} |")
lat=json.loads((R/'latency20ms_diagnostic.json').read_text())['summary']
old=pd.read_csv(ROOT/'results'/'summary.csv');oracle=[]
for name in NAMES:
    for band in ['primary_45_55','diagnostic_30_70']:
        s=old[(old.dataset==name)&(old.scenario==band)&(old.strategy=='oracle_all_times')].iloc[0]
        oracle.append(f"- {NAMES[name]}, {band}: {money(s['net'])}.")
report=f'''# Maker v2 and paired-taker arbitrage: completed replay

**The raw data shows predictive information about order-flow timing. It does not establish robust maker profitability.** The strict 45–55% maker placed no trades; the wider-band MLB dynamic diagnostic made $0.0832 after fees from five filled episodes. One $0.1819 gain accounts for more than the entire net profit.

## What was added

The model now forecasts separate buy and sell waiting distributions, the next opposite-side take, a joint next-side/size-bin distribution, side-specific size distributions, and cumulative buy/sell volume at 1/2/5/10/20/50 trades. It learns empirical queue-fill outcome/time distributions from completed counterfactual quote episodes. The maker can cancel/replace quotes as predictions change, loses queue priority when repricing, reassesses the remaining inventory, and exits when the policy detects adverse flow or the odds leave the permitted band.

These are expanding empirical distributions. No regression or parameter fit replaces the counts. Prices, features and updates are causal, and extending a dataset into the future was tested not to change earlier forecasts. A label enters the distribution only after its entire observation window has matured, with strictly earlier receipt time than the current decision. Flow initialization is 100 completed windows; maker initialization is 30 completed quote episodes per quote placement. The context is known at decision time; later imbalance changes enter later decisions.

## Forward accuracy

All-odds numbers below score valid continuous recording windows. They are **not restricted to 45–55%**, and they are not executable-trade directional accuracy. Forecasts occur every fifth actual trade. The target next side means the very next subsequent trade.

{chr(10).join(flowtable)}

The full model beats unconditional frequencies on raw MLB and NFL, but **loses to simple side persistence on MLB**. This prevents the 75.8% headline from being presented as a superior new predictor. In the actual 45–55% MLB regime, full and unconditional next-side accuracy were both 90.7% on 535 forecasts. Opposite-within-five accuracy improved from 69.0% to 74.8% there. There are no 45–55% NFL observations. Sparse BUF–JAC figures are diagnostics from selected intact short stretches, not independent clean validation.

{chr(10).join(timing)}

“Opposite ≤5” asks whether at least one opposite-aggressor trade appears in the next five trades. Accuracy uses a 50% decision threshold. Timing error is the error in the predicted restricted mean wait, capped at 50; no observed arrival before the episode stop receives the 50-trade penalty. It is not a promise of eventual execution. Size bins are <1, [1,10), [10,100), [100,1000), and ≥100 contracts. Midpoint direction excludes no-change outcomes and means the next change visible at a trade observation, not necessarily every intratrade book flicker.

Full JSON results additionally contain per-side recall, class-balanced accuracy, probability squared errors, waiting-time CDFs in trades and seconds, side-specific expected volume errors, conditional expected waits, disjoint-window checks and five-minute-block bootstrap intervals. Brier scores are mean squared probability errors: lower is better. Accuracy alone can be high by repeatedly choosing the more common event.

The 50-trade windows overlap. In disjoint raw windows, next-side accuracy was 78.4% versus 68.2% on MLB (305 forecasts), and 86.1% versus 82.9% on NFL (791). These remain two game recordings, not evidence of cross-market robustness.

## Predicting our queue fills is harder

{chr(10).join(fills)}

“Any fill” includes a partial fill. The main 45–55% MLB quote sample was small and poorly predicted. In the wider diagnostic, extra conditioning helped MLB any-fill accuracy but hurt NFL. Average fill-rate calibration is not the same as identifying which individual quotes fill. Both-full-pair accuracy of 81.7% MLB / 91.2% NFL in the wide band merely matched always predicting “not both”; it is not an edge.

Quote labels are hypothetical fills under the queue model, not actual private order fills. The learner uses all available matured eligible quote episodes from the original 30–70% and 45–55% shadow runs, deduplicated by timestamp and placement. This expands the primary regime's initialization information, which is why its scored-quote count differs from v1. Flow learning uses all valid odds. Neither dataset eligibility nor features are derived from future ranks/quantiles.

## PnL, $5 initial bankroll per contract

{chr(10).join(pnlt)}

The primary band remains 45–55%; 30–70% is a separately declared diagnostic. One contract per side is used to isolate the signal. Quotes have a maximum 50-trade lifetime and a 300-second stale-order guard. A dynamic policy can cancel earlier for negative updated expected edge, low opposite-flow probability, adverse midpoint probability, or leaving the odds band.

The MLB dynamic diagnostic has two fully paired wins and three one-sided losing exits. The largest win was buying YES at $0.52 and NO at $0.29, giving $0.19 gross / $0.1819 net. Excluding that one episode leaves −$0.0987. The single 20 ms execution sensitivity produced {money(lat['net'])} from {lat['filled']} filled episodes; this is not a replacement for the requested 5 ms baseline or a latency optimum. It changes which cancellations/fills occur and the sample is much too small for a robustness claim.

The v2 baseline differs slightly from v1 because candidates are now checked every fifth trade globally, accounting supports changing quote prices, and fee-plus-position-cost rounding handles fractional fills. No previous result is silently substituted. The fee correction changes the wide MLB static result by five thousandths of a cent.

Historical fixed-quote oracles, supplied as opportunity context (their candidate grid/management differs from v2's dynamic policy):
{chr(10).join(oracle)}

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
- Every closed ledger was independently reconciled from fill cashflows, paired offsets and fees: **{checked} passed**. Thirteen deterministic tests passed, including future-prefix invariance, censor handling, partial-fill basis, cancellation races, queue resets, fractional rounding, delayed exits and two-leg arbitrage failure.
- Models condition on odds and recent state, imbalance, separate taker count/volume shares and arrival rates, ages since each side's last trade, the last three sides, spread and depth/volume ratios. Full contexts back off through odds+last-side and imbalance+last-midpoint-state cells to all history with strength 20. Therefore the “imb” ablation includes midpoint state as well as imbalance; it does not isolate imbalance alone.
- Quote decisions use a probability mixture of none / YES-only / NO-only / both / partial-pair outcomes with conditional net returns. The planned paired spread is substituted for that outcome at the current quote. This remains dependent on estimated one-sided exit losses, not merely a high probability that some trade arrives. Dynamic post-fill cancellation uses the needed taker-side arrival CDF and midpoint risk; it is not an exact exchange-queue hazard model.
- Sparse BUF–JAC has large gaps; its two contracts are the same game. Reported sparse losses and drawdowns cannot bound unobserved fills during gaps. GOOG lacks a trade tape and NSE samples lack precise trade timestamps for this same latency/queue test; they are excluded rather than given invented maker fills.
- Market-by-price data cannot reveal exact cancellations ahead, hidden orders, own impact or matching-engine order sequence. Only live own-order records can validate actual fill probabilities. No claim is made that this is an error-free exchange emulator, that the 50/50 odds band is intrinsically safe, or that the model is ready for unattended real-money trading.

## Reproduction

Install root requirements. Keep the original raw files under `inputs/` and the included v1 shadow labels under `results/`. Run `python -m unittest discover -s maker_v2 -p 'test_v2.py'`, `python maker_v2/run.py`, `python maker_v2/latency_check.py`, then `python maker_v2/report.py`. `run.py` also accepts dataset names to resume selected assets. The bundle contains source, compact frozen side/size/midpoint/opposite-wait forecasts, ledgers, scores, calibration, input hashes and the legacy labels needed for reproduction; original input bytes are already in the earlier datasets and are not duplicated.

Official references: [Kalshi order-book representation](https://docs.kalshi.com/getting_started/orderbook_responses); [fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf). These establish market conventions and fee assumptions, not the validity of hypothetical fills.
'''
(HERE/'maker_v2_report.md').write_text(report)
# Retain the exact implementation hashes independently of the archive container.
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
(HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
bundle_tmp=Path(tempfile.mktemp(suffix='.zip'))
with zipfile.ZipFile(bundle_tmp,'w',zipfile.ZIP_DEFLATED) as z:
    for name in ['maker_replay.py','audit_raw.py','requirements.txt','input_manifest.json','test_replay.py']:
        p=ROOT/name;z.write(p,name)
    for p in (ROOT/'results').iterdir():
        if p.is_file():z.write(p,str(p.relative_to(ROOT)))
    for p in HERE.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.zip','.log'] and not any(p.name.endswith(s) for s in ['_flow_predictions.jsonl.gz','_issued_predictions.jsonl.gz']):
            z.write(p,str(p.relative_to(ROOT)))
    z.write(HERE/'validation.log','maker_v2/validation.log')
os.replace(bundle_tmp,HERE/'maker_v2_reproducible.zip')
with zipfile.ZipFile(HERE/'maker_v2_reproducible.zip') as z:assert z.testzip() is None
print('Saved report, accuracy CSV, figure, source manifest and reproducible bundle.')
