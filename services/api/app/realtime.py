"""Near-real-time A-share market data for the demo (display layer only).

Sources, in fallback order:
  quotes : Tencent (qt.gtimg.cn)  -> Sina (hq.sinajs.cn)
  minute : Tencent minute query   -> pytdx (TDX servers, if reachable) -> AkShare/Sina 1-min bars
  daily K: Tencent fqkline (forward-adjusted)

These are free, unofficial endpoints: rate-limited here, cached briefly, and labelled
"may be delayed" in the UI. The risk engine never uses them — it uses official closes.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo

import httpx

log = logging.getLogger("realtime")
CN = ZoneInfo("Asia/Shanghai")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
INDICES = {"000300.SH": "sh000300", "000905.SH": "sh000905", "000001.SH": "sh000001", "399001.SZ": "sz399001", "399006.SZ": "sz399006"}


def to_symbol(code: str) -> str:
    """'600519.SH' -> 'sh600519' (indices map explicitly because 000300 is not a SZ stock)."""
    if code in INDICES:
        return INDICES[code]
    num, ex = code.split(".")
    return f"{ex.lower()}{num}"


def from_symbol(sym: str) -> str:
    for k, v in INDICES.items():
        if v == sym:
            return k
    return f"{sym[2:]}.{sym[:2].upper()}"


def _f(x: str) -> float | None:
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- session
def session_status(now: dt.datetime | None = None) -> dict[str, str]:
    now = (now or dt.datetime.now(CN)).astimezone(CN)
    hm = now.hour * 100 + now.minute
    if now.weekday() >= 5:
        return {"code": "closed", "label": "休市"}
    if 915 <= hm < 925:
        return {"code": "auction", "label": "集合竞价"}
    if 930 <= hm < 1130 or 1300 <= hm < 1457:
        return {"code": "trading", "label": "交易中"}
    if 1457 <= hm < 1500:
        return {"code": "auction", "label": "收盘竞价"}
    if 1130 <= hm < 1300:
        return {"code": "break", "label": "午间休市"}
    if hm < 915:
        return {"code": "pre", "label": "盘前"}
    return {"code": "closed", "label": "已收盘"}


# ----------------------------------------------------------------------------- parsing
def parse_tencent(text: str) -> dict[str, dict[str, Any]]:
    out = {}
    for line in text.strip().split(";"):
        if "=" not in line or '"' not in line:
            continue
        sym = line.split("=")[0].strip().replace("v_", "")
        f = line.split('"')[1].split("~")
        if len(f) < 49:
            continue
        price, prev = _f(f[3]), _f(f[4])
        ts = f[30]
        out[from_symbol(sym)] = {
            "code": from_symbol(sym), "name": f[1], "price": price, "prev_close": prev, "open": _f(f[5]),
            "high": _f(f[33]), "low": _f(f[34]), "change": _f(f[31]), "pct": (_f(f[32]) or 0) / 100,
            "volume": (_f(f[6]) or 0) * 100, "amount": (_f(f[37]) or 0) * 1e4, "turnover": (_f(f[38]) or 0) / 100,
            "pe": _f(f[39]), "pb": _f(f[46]), "float_mcap": (_f(f[44]) or 0) * 1e8,
            "limit_up": _f(f[47]) if _f(f[47]) and _f(f[47]) > 0 else None,
            "limit_down": _f(f[48]) if _f(f[48]) and _f(f[48]) > 0 else None,
            "bids": [[_f(f[9 + 2 * i]), (_f(f[10 + 2 * i]) or 0) * 100] for i in range(5)],
            "asks": [[_f(f[19 + 2 * i]), (_f(f[20 + 2 * i]) or 0) * 100] for i in range(5)],
            "time": f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[8:10]}:{ts[10:12]}:{ts[12:14]}" if len(ts) >= 14 else ts,
            "source": "tencent",
        }
        q = out[from_symbol(sym)]
        # no trade today: no opening price and zero volume (only meaningful once the session has opened)
        q["no_trade"] = not q["open"] and not q["volume"]
        q["at_limit_up"] = bool(q["limit_up"] and price and price >= q["limit_up"] - 1e-6)
        q["at_limit_down"] = bool(q["limit_down"] and price and price <= q["limit_down"] + 1e-6)
    return out


def parse_sina(text: str) -> dict[str, dict[str, Any]]:
    out = {}
    for line in text.strip().split(";"):
        if "hq_str_" not in line or '"' not in line:
            continue
        sym = line.split("hq_str_")[1].split("=")[0]
        f = line.split('"')[1].split(",")
        if len(f) < 32:
            continue
        price, prev = _f(f[3]), _f(f[2])
        out[from_symbol(sym)] = {
            "code": from_symbol(sym), "name": f[0], "price": price, "prev_close": prev, "open": _f(f[1]),
            "high": _f(f[4]), "low": _f(f[5]), "change": (price - prev) if price and prev else None,
            "pct": (price / prev - 1) if price and prev else 0.0, "volume": _f(f[8]), "amount": _f(f[9]),
            "bids": [[_f(f[11 + 2 * i]), _f(f[10 + 2 * i])] for i in range(5)],
            "asks": [[_f(f[21 + 2 * i]), _f(f[20 + 2 * i])] for i in range(5)],
            "time": f"{f[30]} {f[31]}", "source": "sina",
        }
    return out


# ----------------------------------------------------------------------------- client
@dataclass
class _Cache:
    ttl: float
    store: dict[str, tuple[float, Any]] = field(default_factory=dict)

    def get(self, key: str):
        hit = self.store.get(key)
        return hit[1] if hit and time.time() - hit[0] < self.ttl else None

    def put(self, key: str, value: Any):
        self.store[key] = (time.time(), value)


class MarketData:
    def __init__(self) -> None:
        self.http = httpx.AsyncClient(timeout=8.0, headers=UA, trust_env=False)
        self.quotes = _Cache(ttl=2.0)
        self.minutes = _Cache(ttl=10.0)
        self.klines = _Cache(ttl=60.0)
        self.health: dict[str, dict[str, Any]] = {}
        self._tdx = None          # (ip, port) once a reachable TDX server is found
        self._tdx_checked = 0.0

    def _ok(self, name: str, ok: bool, msg: str = ""):
        self.health[name] = {"ok": ok, "at": dt.datetime.now(CN).isoformat(timespec="seconds"), "msg": msg[:160]}

    # ---------------- quotes
    async def get_quotes(self, codes: list[str]) -> dict[str, dict[str, Any]]:
        want = [c for c in dict.fromkeys(codes) if c]
        out, miss = {}, []
        for c in want:
            q = self.quotes.get(c)
            (out.__setitem__(c, q) if q else miss.append(c))
        for i in range(0, len(miss), 60):
            chunk = miss[i:i + 60]
            got = await self._tencent(chunk) or await self._sina(chunk)
            for c, q in got.items():
                self.quotes.put(c, q)
                out[c] = q
        return out

    async def _tencent(self, codes: list[str]) -> dict:
        try:
            r = await self.http.get("https://qt.gtimg.cn/q=" + ",".join(to_symbol(c) for c in codes))
            res = parse_tencent(r.content.decode("gbk", errors="ignore"))
            self._ok("tencent", bool(res))
            return res
        except Exception as exc:  # noqa: BLE001
            self._ok("tencent", False, str(exc))
            return {}

    async def _sina(self, codes: list[str]) -> dict:
        try:
            r = await self.http.get("https://hq.sinajs.cn/list=" + ",".join(to_symbol(c) for c in codes),
                                    headers={**UA, "Referer": "https://finance.sina.com.cn"})
            res = parse_sina(r.content.decode("gbk", errors="ignore"))
            self._ok("sina", bool(res))
            return res
        except Exception as exc:  # noqa: BLE001
            self._ok("sina", False, str(exc))
            return {}

    # ---------------- intraday minutes
    async def get_minutes(self, code: str) -> dict[str, Any]:
        hit = self.minutes.get(code)
        if hit:
            return hit
        res = await self._tencent_minutes(code)
        if not res["bars"]:
            res = await asyncio.to_thread(self._tdx_minutes, code)
        if not res["bars"]:
            res = await asyncio.to_thread(self._akshare_minutes, code)
        self.minutes.put(code, res)
        return res

    async def _tencent_minutes(self, code: str) -> dict[str, Any]:
        sym = to_symbol(code)
        try:
            r = await self.http.get(f"https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={sym}")
            d = r.json()["data"][sym]["data"]
            day = d["date"]
            bars, pv, pa = [], 0.0, 0.0
            for row in d["data"]:
                hhmm, price, vol, amt = row.split(" ")[:4]
                if hhmm > "1500":   # after-hours fixed-price session
                    continue
                v, a = float(vol) * 100, float(amt)
                bars.append({"t": f"{day[:4]}-{day[4:6]}-{day[6:]} {hhmm[:2]}:{hhmm[2:]}", "p": float(price),
                             "v": max(v - pv, 0.0), "avg": a / v if v else float(price)})
                pv, pa = v, a
            self._ok("tencent_minute", True)
            return {"code": code, "source": "tencent", "bars": bars}
        except Exception as exc:  # noqa: BLE001
            self._ok("tencent_minute", False, str(exc))
            return {"code": code, "source": "tencent", "bars": []}

    def _tdx_minutes(self, code: str) -> dict[str, Any]:
        """pytdx (optional): only used when a TDX server answers on port 7709."""
        try:
            from pytdx.config.hosts import hq_hosts
            from pytdx.hq import TdxHq_API
        except ImportError:
            return {"code": code, "source": "pytdx", "bars": []}
        api = TdxHq_API(raise_exception=True)
        if self._tdx is None and time.time() - self._tdx_checked > 600:
            self._tdx_checked = time.time()
            for _, ip, port in hq_hosts[:40]:
                try:
                    with api.connect(ip, port, time_out=1.5):
                        self._tdx = (ip, port)
                        break
                except Exception:  # noqa: BLE001
                    continue
            self._ok("pytdx", self._tdx is not None, "no reachable TDX server" if self._tdx is None else f"{self._tdx}")
        if self._tdx is None:
            return {"code": code, "source": "pytdx", "bars": []}
        num, ex = code.split(".")
        try:
            with api.connect(*self._tdx, time_out=3):
                rows = api.get_minute_time_data(1 if ex == "SH" else 0, num) or []
            bars = [{"t": None, "p": float(r["price"]), "v": float(r["vol"]) * 100, "avg": float(r["price"])} for r in rows]
            return {"code": code, "source": "pytdx", "bars": bars}
        except Exception as exc:  # noqa: BLE001
            self._tdx = None
            self._ok("pytdx", False, str(exc))
            return {"code": code, "source": "pytdx", "bars": []}

    def _akshare_minutes(self, code: str) -> dict[str, Any]:
        try:
            import akshare as ak

            df = ak.stock_zh_a_minute(symbol=to_symbol(code), period="1")
            last_day = str(df["day"].iloc[-1])[:10]
            df = df[df["day"].astype(str).str.startswith(last_day)]
            bars, cv, ca = [], 0.0, 0.0
            for _, r in df.iterrows():
                v, p = float(r["volume"]), float(r["close"])
                cv, ca = cv + v, ca + float(r.get("amount", p * v) or p * v)
                bars.append({"t": str(r["day"])[:16], "p": p, "v": v, "avg": ca / cv if cv else p})
            self._ok("akshare_minute", True)
            return {"code": code, "source": "akshare-sina", "bars": bars}
        except Exception as exc:  # noqa: BLE001
            self._ok("akshare_minute", False, str(exc))
            return {"code": code, "source": "akshare-sina", "bars": []}

    # ---------------- daily K-line up to today (forward-adjusted)
    async def get_kline(self, code: str, years: int = 3) -> dict[str, Any]:
        key = f"{code}:{years}"
        hit = self.klines.get(key)
        if hit:
            return hit
        sym = to_symbol(code)
        end = dt.date.today()
        rows: dict[str, list] = {}
        start = end - dt.timedelta(days=365 * years)
        cur = start
        try:
            while cur <= end:  # the endpoint returns at most 640 bars per call
                stop = min(end, cur + dt.timedelta(days=800))
                r = await self.http.get("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                                        params={"param": f"{sym},day,{cur},{stop},640,qfq"})
                d = r.json()["data"][sym]
                for b in d.get("qfqday") or d.get("day") or []:
                    rows[b[0]] = [b[0], float(b[1]), float(b[3]), float(b[4]), float(b[2]), float(b[5]) * 100]
                cur = stop + dt.timedelta(days=1)
            self._ok("tencent_kline", True)
        except Exception as exc:  # noqa: BLE001
            self._ok("tencent_kline", False, str(exc))
        bars = [rows[k] for k in sorted(rows)]       # t, open, high, low, close, volume (shares)
        res = {"code": code, "source": "tencent-qfq", "bars": bars}
        if len(bars) < 30:   # e.g. BSE 920xxx codes: Tencent only has the latest bar
            sina = await self._sina_kline(sym, years)
            if len(sina) > len(bars):
                res = {"code": code, "source": "sina-unadjusted", "bars": sina}
        if res["bars"]:
            self.klines.put(key, res)
        return res

    async def _sina_kline(self, sym: str, years: int) -> list[list]:
        """Sina daily bars (unadjusted; at most 1023 per call); volume in shares."""
        try:
            r = await self.http.get("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData",
                                    params={"symbol": sym, "scale": 240, "ma": "no", "datalen": min(1023, 250 * years)},
                                    headers={"Referer": "https://finance.sina.com.cn"})
            data = r.json() or []
            self._ok("sina_kline", True)
            return [[b["day"], float(b["open"]), float(b["high"]), float(b["low"]), float(b["close"]), float(b["volume"])]
                    for b in data]
        except Exception as exc:  # noqa: BLE001
            self._ok("sina_kline", False, str(exc))
            return []
