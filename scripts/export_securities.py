"""Export the full A-share securities list for the web search -> web/public/demo-data/securities_all.json

    python scripts/ingest.py --steps basic      # once: BaoStock stock_basic + BSE list
    python scripts/export_securities.py

Rows: code, name, pinyin initials (variants for heteronyms, '|'-separated), SW L1 industry, board,
and whether the stock is in the research universe (historical CSI300/CSI500 members), so the UI
knows when factor data exists.
"""
from __future__ import annotations

import itertools
import json
import logging
import sys

import pandas as pd
from pypinyin import Style, pinyin

from quantkit.config import load_settings
from quantkit.data.codes import board_of
from quantkit.data.store import DataStore
from quantkit.data.sw import sw_name

log = logging.getLogger("securities")
INDICES = [("000300.SH", "沪深300"), ("000905.SH", "中证500"), ("000001.SH", "上证指数"),
           ("399001.SZ", "深证成指"), ("399006.SZ", "创业板指")]


def initials(name: str, max_variants: int = 4) -> str:
    """'贵州茅台' -> 'gzmt'; heteronyms add variants ('长江电力' -> 'cjdl|zjdl'); letters/digits kept."""
    per_char = []
    for ch in name.replace("Ａ", "A").replace("Ｂ", "B"):
        if ch.isascii():
            if ch.isalnum():
                per_char.append([ch.lower()])
            continue
        opts = pinyin(ch, style=Style.FIRST_LETTER, heteronym=True, errors="ignore")
        letters = list(dict.fromkeys(o[0].lower() for o in (opts[0] if opts else []) if o)) or []
        if letters:
            per_char.append(letters[:2])
    # the phrase-aware reading first (pypinyin picks it without heteronym=True)
    best = "".join(x[0] for x in pinyin(name, style=Style.FIRST_LETTER, errors=lambda s: [c.lower() for c in s if c.isalnum()]) if x).lower()
    variants = [best] + ["".join(v) for v in itertools.islice(itertools.product(*per_char), 16)]
    return "|".join(list(dict.fromkeys(v for v in variants if v))[:max_variants])


def _mcap(rs: dict | None) -> float | None:
    v = rs.get("mcap") if rs else None
    return round(float(v), 2) if isinstance(v, (int, float)) and v == v else None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    cfg = load_settings()
    store = DataStore(cfg.data_dir)
    out = cfg.root / "web" / "public" / "demo-data"
    basic = store.read("raw", "stock_basic.parquet")
    basic = basic[(basic["type"] == "1") & (basic["status"] == "1")][["code", "name", "ipo_date"]]
    if store.exists("raw", "stock_basic_bj.parquet"):
        bj = store.read("raw", "stock_basic_bj.parquet")[["code", "name", "ipo_date"]]
        basic = pd.concat([basic, bj[~bj["code"].isin(basic["code"])]], ignore_index=True)
    ind = store.read("curated", "industry.parquet")
    latest = ind.sort_values("start").groupby("code")["sw_l1"].last().map(sw_name)
    research = json.loads((out / "securities.json").read_text(encoding="utf-8"))
    rset = {s["code"]: s for s in research["items"]}

    rows = []
    for r in basic.itertuples(index=False):
        rs = rset.get(r.code)
        rows.append([r.code, r.name.strip(), initials(r.name.strip()), latest.get(r.code, "") or "",
                     board_of(r.code), 1 if rs else 0, _mcap(rs)])
    # research-universe stocks that have since been delisted stay searchable (their data exist)
    known = {x[0] for x in rows}
    for c, rs in rset.items():
        if c not in known:
            rows.append([c, rs["name"], initials(rs["name"]), rs["industry"], board_of(c), 1, _mcap(rs)])
    rows.sort(key=lambda x: x[0])
    idx = [[c, n, initials(n), "指数", "index", 0, None] for c, n in INDICES]
    payload = {"asof": pd.Timestamp.today().strftime("%Y-%m-%d"),
               "fields": ["code", "name", "py", "industry", "board", "research", "mcap"],
               "rows": idx + rows,
               "note": "全部上市 A 股（沪深来自 BaoStock，北交所来自 AkShare）；research=1 表示在研究股票池（沪深300/中证500 历史成分），有因子数据；mcap 为研究数据日的流通市值（亿元）"}
    path = out / "securities_all.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    log.info("wrote %s: %d rows (%.0f KB), %d in research universe", path.name, len(payload["rows"]),
             path.stat().st_size / 1024, sum(x[5] for x in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
