# Dataset provenance

| Dataset | Location | Provenance / scope |
|---|---|---|
| Active ten-market capture | `combined_paper/active_30min_20260929T1640Z/feed.jsonl.gz` | Original public market-data recording, 29 Sep 2026; 418,577 records, 11,248 trade messages; metadata and fee snapshots alongside |
| Earlier live pilots | `historical/Kalshi_Three_Leg_Pilot.zip`, `Kalshi_Recovered_Replay.zip` | Original Sep 28/29 recordings; recovered interrupted capture must retain truncation/censoring flags |
| MLB TEX-SEA | Research and sports ZIPs | CryptoStruct `kalshi-KXMLBGAME-26SEP101610TEXSEA-SEA-2026-09-10.txt.zst` |
| NFL WAS-PHI | Research and sports ZIPs | CryptoStruct `kalshi-KXNFLGAME-26SEP13WASPHI-PHI-2026-09-13.txt.zst` |
| BTC 15-minute contract | `historical/kalshi_research_2026-09-21.zip` | CryptoStruct `kalshi-btc15m-2026-09-11.txt.zst`; historical retained interval 14:00:02-14:14:00 UTC |
| Broad order-book shard | Same research ZIP | `hf-kalshi-orderbooks.jsonl.gz`; source cited by original report: Hugging Face `lerchen3/kalshi-orderbook-alpha`; not an aligned trade tape |
| BUF/JAC NFL | `data/orderbook.parquet`, `data/trades.parquet` | Recovered original files; earlier audit reports 8,047 books / 111,160 trades. Counts checked anew in portfolio inventory if supported by installed parquet reader; original acquisition URL not recovered |
| External sports validation | `historical/kalshi_external_sports_validation.zip` | Derived external books/predictions; exact upstream rights and native coverage not independently reconstructed |

Sources cited in original artifacts: https://cryptostruct.com/download and https://huggingface.co/datasets/lerchen3/kalshi-orderbook-alpha . Metadata is provenance, not a new validation of provider accuracy or redistribution permission.

Raw files are staged for a **private** GitHub upload. Provider redistribution terms have not been established from the recovered artifacts. Preserve attribution and do not assume public availability grants an open-source data license. Large datasets use the supplied Git LFS patterns; hashes are in `MANIFEST.json` and archived source manifests. Duplicate source coverage across ZIPs is intentional preservation, not independent observations.
