from pathlib import Path
import shutil,subprocess,sys
root=Path(__file__).resolve().parent
src=root/'combined_paper/active_30min_20260929T1640Z';dst=root/'verification/replayed'
dst.mkdir(parents=True,exist_ok=True)
for name in ['capture_summary.json','feed.jsonl.gz']:shutil.copy2(src/name,dst/name)
with (root/'verification/replay.log').open('w') as f:
 subprocess.run([sys.executable,str(root/'combined_paper/replay_combined.py'),str(dst)],stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(root/'verify_results.py'),str(dst)],check=True)
