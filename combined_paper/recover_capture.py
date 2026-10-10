"""Recover complete JSONL records from an interrupted gzip into a NEW directory.

Preserves the original compressed bytes and intended duration. Never invents end
metadata, a graceful stop, or missing market messages. No network or credentials.
"""
import argparse,gzip,hashlib,json,zlib
from pathlib import Path
from collections import Counter

def recover(source,dest):
    source,dest=Path(source),Path(dest)
    dest.mkdir(parents=True,exist_ok=False)
    decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
    pending=b'';first=last=None;counts=Counter();n=0;digest=hashlib.sha256()
    with source.open('rb') as f,gzip.open(dest/'feed.jsonl.gz','wt') as out:
        while True:
            block=f.read(65536)
            if not block:break
            digest.update(block)
            pending+=decoder.decompress(block)
            while b'\n' in pending:
                line,pending=pending.split(b'\n',1)
                if not line.strip():continue
                row=json.loads(line)
                if first is None:first=row
                if last is not None and row['t']<last['t']:
                    raise ValueError('Nonmonotone capture clock')
                last=row;n+=1
                counts[row.get('data',{}).get('type',row['kind'])]+=1
                out.write(line.decode('utf-8')+'\n')
    if first is None or first['kind']!='metadata':raise ValueError('Missing initial metadata')
    if decoder.eof:raise ValueError('Source has a complete gzip ending; use normal capture handling')
    summary=dict(incomplete=True,start_wall=first['wall'],end_wall=last['wall'],
        metadata=first['metadata'],end_metadata=None,messages=dict(counts),
        planned_duration_seconds=first['duration'],recovered_seconds=last['t']-first['t'],
        recovery=dict(source_path=str(source.resolve()),source_sha256=digest.hexdigest(),
                      complete_records=n,trailing_incomplete_bytes_discarded=len(pending),
                      gzip_footer_missing=True),
        note='Only complete messages recovered. No graceful stop or terminal fee check recorded.')
    (dest/'capture_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('metadata','end_metadata')},indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args()
    recover(a.source,a.destination)
