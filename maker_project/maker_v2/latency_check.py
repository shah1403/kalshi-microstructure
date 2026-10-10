"""One execution sensitivity for the small positive diagnostic; no selection."""
from run import *
source=lambda:raw_events(next((ROOT/'inputs').glob('*MLB*.zst')))
rows,_=extract(source)
with gzip.open(OUT/'KXMLBGAME_MLB_issued_predictions.jsonl.gz','rt') as f:
    forecasts={r['i']:r for r in map(json.loads,f)}
c=Config(latency_ms=20.)
s,l=strategy(source,rows,forecasts,ROOT,'KXMLBGAME_MLB',(.3,.7),'probability_dynamic',False,c)
s['latency_ms']=20.
(OUT/'latency20ms_diagnostic.json').write_text(json.dumps(dict(summary=s,ledger=l)))
print(json.dumps(s))
