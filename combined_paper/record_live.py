"""Bounded, market-data-only recorder. Credentials enter stdin, never the log."""
import argparse, asyncio, gzip, json, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'kalshi_paper'))
from paper_engine import WS, auth_headers, market_metadata

async def record(args, credentials):
    import websockets
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    key = load_pem_private_key(credentials['pem'].encode(), password=None)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    async def fetch_metadata():
        limit = asyncio.Semaphore(4)
        async def one(ticker):
            async with limit:
                market, series = await asyncio.to_thread(market_metadata, ticker)
                return ticker, dict(market=market, series=series)
        return dict(await asyncio.gather(*(one(t) for t in args.tickers)))
    metadata = await fetch_metadata()
    counts = Counter()
    started = time.time()
    def row(kind, **kw):
        return dict(kind=kind, t=time.monotonic(), wall=time.time(), **kw)
    with gzip.open(output / 'feed.jsonl.gz', 'xt') as log:
        def emit(kind, **kw):
            log.write(json.dumps(row(kind, **kw)) + '\n')
            log.flush()
        emit('metadata', metadata=metadata, duration=args.seconds)
        try:
            async with websockets.connect(WS, additional_headers=auth_headers(credentials['id'], key),
                                          open_timeout=15, ping_interval=10, ping_timeout=10,
                                          max_queue=1024) as ws:
                for i, channel in enumerate(('orderbook_delta', 'trade'), 1):
                    await ws.send(json.dumps(dict(id=i, cmd='subscribe', params=dict(
                        channels=[channel], market_tickers=args.tickers))))
                await ws.send(json.dumps(dict(id=3, cmd='subscribe', params=dict(
                    channels=['market_lifecycle_v2']))))
                print('Authenticated data recording started; no order endpoints exist in this program.', flush=True)
                deadline = time.monotonic() + args.seconds
                next_status = time.monotonic() + 30
                while time.monotonic() < deadline:
                    try:
                        data = json.loads(await asyncio.wait_for(ws.recv(), timeout=.5))
                    except asyncio.TimeoutError:
                        emit('timer')
                        continue
                    emit('message', data=data)
                    counts[data.get('type', 'unknown')] += 1
                    if data.get('type') == 'error':
                        raise RuntimeError('Subscription rejected: ' + str(data.get('msg', {}).get('code')))
                    if time.monotonic() >= next_status:
                        print(json.dumps(dict(elapsed_s=round(time.time()-started), messages=dict(counts))), flush=True)
                        next_status = time.monotonic() + 30
                emit('end', reason='bounded_recording_complete')
        except Exception as exc:
            emit('gap', reason=type(exc).__name__)
            print('Recording stopped:', type(exc).__name__, flush=True)
            raise
    # Re-check fee metadata; a change invalidates the pilot rather than silently changing charges.
    end_metadata = await fetch_metadata()
    summary = dict(start_wall=started, end_wall=time.time(), messages=dict(counts),
                   metadata=metadata, end_metadata=end_metadata, credentials_saved=False)
    (output/'capture_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(completed=True, messages=dict(counts))), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('tickers', nargs='+')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800:
        parser.error('Use a bounded 1–1800 second capture')
    asyncio.run(record(args, json.load(sys.stdin)))
