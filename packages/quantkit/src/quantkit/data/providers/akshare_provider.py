"""AkShare provider: SW industry history, point-in-time financial cross-sections, index bars."""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

import pandas as pd
import requests

from ..codes import to_canonical
from .base import ensure_socket_timeout, retry

log = logging.getLogger("quantkit.data")

SW_CLASS_URL = "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"
BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/124 Safari/537.36",
                   "Referer": "https://www.swsresearch.com/"}
CERT_DIR = Path(__file__).resolve().parent.parent / "certs"


def ca_bundle() -> str:
    """certifi roots + intermediates shipped in ``data/certs`` (for servers with broken chains)."""
    import certifi

    out = Path(tempfile.gettempdir()) / "quantkit_ca_bundle.pem"
    parts = [Path(certifi.where()).read_text()] + [p.read_text() for p in sorted(CERT_DIR.glob("*.pem"))]
    out.write_text("\n".join(parts))
    return str(out)

# Cross-sectional report tables published by Eastmoney's data center; each row carries the
# announcement date, which makes them usable point-in-time.
FIN_TABLES = {
    "yjbb": {  # 业绩报表
        "fn": "stock_yjbb_em",
        "cols": {"股票代码": "code", "最新公告日期": "ann_date", "每股收益": "eps",
                 "营业总收入-营业总收入": "revenue", "营业总收入-同比增长": "revenue_yoy",
                 "净利润-净利润": "net_profit", "净利润-同比增长": "profit_yoy",
                 "每股净资产": "bvps", "净资产收益率": "roe", "每股经营现金流量": "cfps",
                 "销售毛利率": "gross_margin"},
    },
    "zcfz": {  # 资产负债表
        "fn": "stock_zcfz_em",
        "cols": {"股票代码": "code", "公告日期": "ann_date", "资产-总资产": "total_assets",
                 "资产-总资产同比": "assets_yoy", "负债-总负债": "total_liab",
                 "资产负债率": "debt_ratio", "股东权益合计": "equity"},
    },
}


class AkShareProvider:
    def __init__(self) -> None:
        ensure_socket_timeout()
        import akshare as ak

        self.ak = ak

    # ------------------------------------------------------------------
    def sw_industry_history(self) -> pd.DataFrame:
        """Shenwan (2021) industry history: one row per (code, effective start date)."""
        try:
            df = self._sw_direct()
        except Exception as exc:  # noqa: BLE001
            log.warning("direct SW download failed (%s); trying akshare", type(exc).__name__)
            df = self._sw_via_akshare()
        df = df.rename(columns={"symbol": "code", "start_date": "start", "industry_code": "sw_code"})
        df["code"] = df["code"].astype(str).str.zfill(6).map(to_canonical)
        df["start"] = pd.to_datetime(df["start"])
        df["sw_code"] = df["sw_code"].astype(str).str.zfill(6)
        df["sw_l1"] = df["sw_code"].str[:2]  # level-1 group id
        return df[["code", "start", "sw_code", "sw_l1"]].sort_values(["code", "start"]).reset_index(drop=True)

    @retry(times=3)
    def _sw_via_akshare(self) -> pd.DataFrame:
        return self.ak.stock_industry_clf_hist_sw()

    @retry(times=3)
    def _sw_direct(self) -> pd.DataFrame:
        # The SW server omits its intermediate certificate, which breaks verification in
        # clients that do not fetch AIA intermediates (Python). We ship that intermediate
        # (GeoTrust G2 TLS CN RSA4096 SHA256 2022 CA1, signed by DigiCert Global Root G2)
        # and verify against certifi + it: full TLS verification stays on.
        r = requests.get(SW_CLASS_URL, timeout=60, headers=BROWSER_HEADERS, verify=ca_bundle())
        r.raise_for_status()
        raw = pd.read_excel(io.BytesIO(r.content), dtype={"股票代码": str, "行业代码": str})
        return raw.rename(columns={"股票代码": "symbol", "计入日期": "start_date", "行业代码": "industry_code"})

    # ------------------------------------------------------------------
    @retry(times=4, wait=3.0)
    def financial_cross_section(self, kind: str, period: str) -> pd.DataFrame:
        """All stocks' report for ``period`` (YYYYMMDD quarter end) with announcement dates."""
        spec = FIN_TABLES[kind]
        df = getattr(self.ak, spec["fn"])(date=period)
        if df is None or df.empty:
            return pd.DataFrame(columns=["code", "period", "ann_date", *list(spec["cols"].values())[2:]])
        df = df.rename(columns=spec["cols"])[list(spec["cols"].values())]
        df["code"] = df["code"].astype(str).str.zfill(6).map(to_canonical)
        df["ann_date"] = pd.to_datetime(df["ann_date"], errors="coerce")
        df["period"] = pd.to_datetime(period)
        for c in df.columns:
            if c not in ("code", "ann_date", "period"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    @retry(times=4, wait=3.0)
    def daily_bars_sina(self, code: str, start: str, end: str) -> pd.DataFrame:
        """Fallback daily bars from Sina in the BaoStock schema.

        Returns are total returns from the back-adjustment factor; ``preclose`` is the
        ex-rights reference price implied by them. Suspended days are simply absent here
        (the panel builder reindexes to the calendar and marks them ``tradestatus=0``).
        ST flags and valuation ratios are not available from this source (left NaN).
        """
        num, ex = to_canonical(code).split(".")
        sym = f"{ex.lower()}{num}"
        s, e = start.replace("-", ""), end.replace("-", "")
        raw = self.ak.stock_zh_a_daily(symbol=sym, start_date=s, end_date=e, adjust="")
        if raw is None or raw.empty:
            return pd.DataFrame()
        hfq = self.ak.stock_zh_a_daily(symbol=sym, start_date=s, end_date=e, adjust="hfq")
        raw["date"] = pd.to_datetime(raw["date"])
        hfq["date"] = pd.to_datetime(hfq["date"])
        df = raw.merge(hfq[["date", "close"]].rename(columns={"close": "close_hfq"}), on="date", how="left")
        df = df.sort_values("date")
        df["ret"] = df["close_hfq"].pct_change()
        df["preclose"] = df["close"] / (1.0 + df["ret"])
        out = pd.DataFrame({
            "date": df["date"], "code": to_canonical(code),
            "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
            "preclose": df["preclose"], "volume": df["volume"], "amount": df["amount"],
            "turn": df["turnover"] * 100.0,  # BaoStock reports turnover in percent
            "tradestatus": (df["volume"] > 0).astype("int8"),
            "isST": pd.Series(pd.NA, index=df.index, dtype="Int8"),
            "peTTM": float("nan"), "pbMRQ": float("nan"), "psTTM": float("nan"), "pcfNcfTTM": float("nan"),
            "ret": df["ret"],
            "float_shares": df["outstanding_share"],
        })
        return out.reset_index(drop=True)

    @retry(times=4, wait=3.0)
    def disclosure_dates(self, period: str) -> pd.DataFrame:
        """Actual first-disclosure date of each company's periodic report for ``period``."""
        df = self.ak.stock_yysj_em(symbol="沪深A股", date=period)
        if df is None or df.empty:
            return pd.DataFrame(columns=["code", "period", "disclosed"])
        return pd.DataFrame({"code": df["股票代码"].astype(str).str.zfill(6).map(to_canonical),
                             "period": pd.to_datetime(period),
                             "disclosed": pd.to_datetime(df["实际披露时间"], errors="coerce")})

    @retry()
    def bse_stock_list(self) -> pd.DataFrame:
        """Listed Beijing Stock Exchange stocks (BaoStock does not cover the BSE)."""
        import akshare as ak

        df = ak.stock_info_bj_name_code()
        out = pd.DataFrame({"code": df["证券代码"].astype(str).str.zfill(6).map(to_canonical), "name": df["证券简称"],
                            "ipo_date": pd.to_datetime(df["上市日期"], errors="coerce"), "csrc_industry": df["所属行业"]})
        out["type"], out["status"] = "1", "1"
        return out

    @retry()
    def current_index_members(self, index: str = "000300") -> pd.DataFrame:
        """*Current* constituents from CSIndex. Using them for history introduces survivorship
        bias — only for smoke tests when historical constituents are unavailable."""
        df = self.ak.index_stock_cons_csindex(symbol=index)
        return pd.DataFrame({"snapshot": pd.Timestamp("1990-01-01"),
                             "code": df["成分券代码"].astype(str).str.zfill(6).map(to_canonical),
                             "name": df["成分券名称"], "index": f"csi{index}_current"})

    @retry()
    def index_daily(self, symbol: str = "sh000300") -> pd.DataFrame:
        df = self.ak.stock_zh_index_daily(symbol=symbol)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["ret"] = df["close"].pct_change()
        df["symbol"] = symbol
        return df[["date", "symbol", "open", "high", "low", "close", "volume", "ret"]]
