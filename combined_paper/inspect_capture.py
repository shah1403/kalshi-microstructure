"""Read-only status of a possibly still-open gzip recording; no API calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import zlib


def inspect(directory):
    directory = Path(directory)
    feed = directory / 'feed.jsonl.gz'
    if not feed.exists():
        return {'state': 'waiting_for_metadata', 'bytes': 0}
    raw = zlib.decompressobj(31).decompress(feed.read_bytes())
    lines = raw.split(b'\n')[:-1]
    counts, trades, snapshots = Counter(), Counter(), set()
    first = last = None
    end = False
    for line in lines:
        row = json.loads(line)
        first = row['wall'] if first is None else first
        last = row['wall']
        if row['kind'] == 'message':
            message = row['data']
            kind = message.get('type', 'unknown')
            counts[kind] += 1
            ticker = message.get('msg', {}).get('market_ticker')
            if kind == 'trade':
                trades[ticker] += 1
            if kind == 'orderbook_snapshot':
                snapshots.add(ticker)
        if row['kind'] == 'end':
            end = True
    return dict(records=len(lines), bytes=feed.stat().st_size,
                elapsed_s=round(last-first, 1) if first is not None else None,
                record_age_s=round(time.time()-last, 1) if last is not None else None,
                messages=dict(counts), trades_by_market=dict(trades),
                snapshot_markets=len(snapshots), completed_capture=end)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    print(json.dumps(inspect(args.directory), indent=2))
