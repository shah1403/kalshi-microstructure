"""Reconcile a fresh replay with the preserved results and independently count raw messages."""
from pathlib import Path
from collections import Counter
import gzip,json,sys,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent
original=ROOT/'combined_paper/active_30min_20260929T1640Z'
rerun=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'verification/replayed'
a=json.loads((original/'combined_results.json').read_text());b=json.loads((rerun/'combined_results.json').read_text())
assert a==b,'Replay output differs from historical combined_results.json'
counts=Counter();n=0
with gzip.open(original/'feed.jsonl.gz','rt') as f:
 for line in f:
  row=json.loads(line);n+=1;counts[row.get('data',{}).get('type',row.get('kind','unknown'))]+=1
markets=list(b['markets'].values());weighted={}
for level in ['all','full']:
 vals=[m['flow_accuracy'][level] for m in markets if level in m['flow_accuracy']]
 den=sum(v['n'] for v in vals)
 weighted[level]={key:sum(v['n']*v[key] for v in vals)/den for key in ['next_side_accuracy','opposite_within_five_accuracy']}
 weighted[level]['n']=den
sh=[]
for f in rerun.glob('*/maker.json'):sh.extend(json.loads(f.read_text())['shadows'])
forecasts=[]
for f in rerun.glob('*/ledger.json'):
 d=json.loads(f.read_text())
 # Canonical forecast rows are exported separately as CSV; ledger shape is retained unchanged.
paired=sum(x['paired'] for x in sh);filled=sum(x['filled'] for x in sh)
gross=sum(m['taker_gross']+m['maker_gross'] for m in markets);fees=sum(m['taker_fees']+m['maker_fees'] for m in markets)
assert abs((b['cash']-b['initial_cash'])-(gross-fees))<1e-8
r=dict(raw_records=n,message_types=dict(counts),exact_historical_replay_match=True,flow=weighted,
 shadows=dict(n=len(sh),paired=paired,filled=filled,one_sided=filled-paired,no_fill=len(sh)-filled,completion=paired/len(sh),diagnostic_net=sum(x['net'] for x in sh)),
 portfolio=dict(initial=b['initial_cash'],ending=b['cash'],gross=gross,fees=fees,net=b['cash']-b['initial_cash'],max_drawdown=b['max_drawdown_observed'],funded_taker_trades=sum(m['taker_closed'] for m in markets),funded_maker_trades=sum(m['maker_closed'] for m in markets),arbitrage_trades=sum(m['arbitrage_trades'] for m in markets)),
 feed_sha256=hashlib.file_digest((original/'feed.jsonl.gz').open('rb'),'sha256').hexdigest())
(ROOT/'verification/reconciliation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
