"""Monitoring and push alerts (F10): scheduled rule checks + WeCom / Feishu / DingTalk webhooks.

Rules (all evaluated on the user's lists, which the browser syncs to POST /api/monitor/state):
    vol     holdings' expected annual volatility above a threshold (risk model, research data)
    drift   holdings vs the target portfolio (组合构建 settings): single stock / industry deviation
    regime  CSI300 20-day annualised volatility crosses the high-volatility threshold (state change)
    limit   holdings / watchlist stocks at limit-up, limit-down or suspended (live quotes)
    data    a market-data source failing

Schedule: intraday rules (limit, regime) every ``intraday_minutes`` while the market is open;
all rules once after the close (``eod_time``) on weekdays. The same event fires at most once a
day (regime: only when the state changes).

Webhook URLs are kept in state/monitor.json (git-ignored) or in .env (WECOM_WEBHOOK_URL,
FEISHU_WEBHOOK_URL / FEISHU_SECRET, DINGTALK_WEBHOOK_URL / DINGTALK_SECRET). Only the official
webhook hosts are accepted, so the server cannot be pointed at arbitrary URLs.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .realtime import CN, MarketData, session_status

log = logging.getLogger("monitor")
ROOT = Path(__file__).resolve().parents[3]
STATE_FILE = ROOT / "state" / "monitor.json"
ALLOWED_HOSTS = {"wecom": ("qyapi.weixin.qq.com",), "feishu": ("open.feishu.cn", "open.larksuite.com"),
                 "dingtalk": ("oapi.dingtalk.com",)}
CHANNEL_CN = {"wecom": "企业微信", "feishu": "飞书", "dingtalk": "钉钉"}
RULES = {
    "vol": {"name": "组合波动超阈值", "params": {"threshold": 0.15, "model": "lw_nonlinear"}, "when": "收盘后"},
    "drift": {"name": "偏离目标权重", "params": {"stock": 0.03, "industry": 0.05}, "when": "收盘后"},
    "regime": {"name": "市场区制切换", "params": {"threshold": 0.2425}, "when": "盘中 + 收盘后"},
    "limit": {"name": "持仓 / 自选涨跌停、停牌", "params": {"include_watch": True}, "when": "盘中 + 收盘后"},
    "data": {"name": "数据源异常", "params": {}, "when": "盘中 + 收盘后"},
}
INTRADAY = ("limit", "regime", "data")
DEFAULT = {
    "rules": {k: {"enabled": k != "drift", "params": copy.deepcopy(v["params"])} for k, v in RULES.items()},
    "channels": {c: {"enabled": False, "url": "", "secret": ""} for c in CHANNEL_CN},
    "schedule": {"enabled": True, "intraday_minutes": 5, "eod_time": "15:10"},
}


class MonitorError(ValueError):
    pass


def validate_url(channel: str, url: str) -> str:
    url = url.strip()
    if not url:
        return ""
    u = urllib.parse.urlparse(url)
    if u.scheme != "https" or u.hostname not in ALLOWED_HOSTS[channel]:
        raise MonitorError(f"{CHANNEL_CN[channel]} webhook 必须是 https://{' 或 '.join(ALLOWED_HOSTS[channel])} 下的地址")
    return url


def mask(url: str) -> str:
    return url[:38] + "…" + url[-4:] if len(url) > 48 else ("已设置" if url else "")


# ----------------------------------------------------------------------------- message formats
def payload(channel: str, title: str, body: str, level: str = "warn", secret: str = "") -> tuple[dict, dict]:
    """(json body, extra query params) for one webhook message."""
    tag = {"critical": "🔴", "warn": "🟠", "info": "🔵"}.get(level, "🔵")
    stamp = dt.datetime.now(CN).strftime("%Y-%m-%d %H:%M")
    footer = f"DeFaCo Studio · {stamp} · 仅供研究与教学，不构成投资建议"
    if channel == "wecom":
        color = "warning" if level != "info" else "info"
        return {"msgtype": "markdown", "markdown": {"content": f"### {tag} <font color=\"{color}\">{title}</font>\n{body}\n\n<font color=\"comment\">{footer}</font>"}}, {}
    if channel == "feishu":
        msg: dict[str, Any] = {"msg_type": "interactive", "card": {
            "header": {"title": {"tag": "plain_text", "content": f"{tag} {title}"}, "template": "orange" if level != "info" else "blue"},
            "elements": [{"tag": "markdown", "content": body}, {"tag": "note", "elements": [{"tag": "plain_text", "content": footer}]}]}}
        if secret:   # Feishu: sign = base64(HMAC-SHA256(key = f"{ts}\n{secret}", msg = ""))
            ts = str(int(time.time()))
            msg["timestamp"] = ts
            msg["sign"] = base64.b64encode(hmac.new(f"{ts}\n{secret}".encode(), b"", hashlib.sha256).digest()).decode()
        return msg, {}
    # dingtalk: sign = urlencode(base64(HMAC-SHA256(key = secret, msg = f"{ts}\n{secret}")))
    params = {}
    if secret:
        ts = str(int(time.time() * 1000))
        params = {"timestamp": ts, "sign": base64.b64encode(hmac.new(secret.encode(), f"{ts}\n{secret}".encode(), hashlib.sha256).digest()).decode()}
    return {"msgtype": "markdown", "markdown": {"title": title, "text": f"### {tag} {title}\n\n{body}\n\n> {footer}"}}, params


def check_response(channel: str, status: int, data: Any) -> str | None:
    """None if delivered, else an error message (the three services report errors in the body)."""
    if status != 200:
        return f"HTTP {status}"
    if not isinstance(data, dict):
        return None
    code = data.get("errcode", data.get("code", data.get("StatusCode", 0)))
    return None if code in (0, None) else f"{data.get('errmsg') or data.get('msg') or data.get('StatusMessage')} (code {code})"


# ----------------------------------------------------------------------------- monitor
class Monitor:
    def __init__(self, md: MarketData, pf_getter, http: httpx.AsyncClient | None = None, state_file: Path = STATE_FILE):
        self.md = md
        self.pf_getter = pf_getter            # () -> PortfolioEngine | None (loaded in the background)
        self.http = http or httpx.AsyncClient(timeout=10.0)
        self.file = state_file
        self.state = self._load()
        self.last_intraday = 0.0
        self.last_run: str | None = None
        self.lock = asyncio.Lock()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> dict:
        try:
            s = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            s = {}
        cfg = copy.deepcopy(DEFAULT)
        for k, v in (s.get("config") or {}).get("rules", {}).items():
            if k in cfg["rules"]:
                cfg["rules"][k]["enabled"] = bool(v.get("enabled", cfg["rules"][k]["enabled"]))
                cfg["rules"][k]["params"].update({p: v["params"][p] for p in v.get("params", {}) if p in cfg["rules"][k]["params"]})
        for k, v in (s.get("config") or {}).get("channels", {}).items():
            if k in cfg["channels"]:
                cfg["channels"][k].update({f: v.get(f, cfg["channels"][k][f]) for f in ("enabled", "url", "secret")})
        cfg["schedule"].update((s.get("config") or {}).get("schedule", {}))
        return {"config": cfg, "client": s.get("client", {}), "events": s.get("events", []),
                "fired": s.get("fired", {}), "regime_state": s.get("regime_state"), "eod_done": s.get("eod_done")}

    def save(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
        os.chmod(tmp, 0o600)          # holds webhook URLs and the user's holdings
        tmp.replace(self.file)

    # ------------------------------------------------------------------ config
    def channel(self, name: str) -> dict:
        """Effective channel config: .env overrides the UI-entered URL."""
        c = dict(self.state["config"]["channels"][name])
        env_url = os.environ.get(f"{name.upper()}_WEBHOOK_URL", "")
        env_secret = os.environ.get(f"{name.upper()}_SECRET", "")
        if env_url:
            c.update(url=env_url, source="env")
        if env_secret:
            c["secret"] = env_secret
        return c

    def public_config(self) -> dict:
        cfg = self.state["config"]
        chans = {}
        for k in CHANNEL_CN:
            c = self.channel(k)
            chans[k] = {"name": CHANNEL_CN[k], "enabled": c["enabled"], "configured": bool(c["url"]), "url_masked": mask(c["url"]),
                        "has_secret": bool(c.get("secret")), "from_env": c.get("source") == "env"}
        return {"rules": {k: {**v, **{"name": RULES[k]["name"], "when": RULES[k]["when"]}} for k, v in cfg["rules"].items()},
                "channels": chans, "schedule": cfg["schedule"], "last_run": self.last_run, "eod_done": self.state.get("eod_done"),
                "client_synced": self.state["client"].get("updated"), "session": session_status()}

    def update_config(self, patch: dict) -> dict:
        cfg = self.state["config"]
        for k, v in (patch.get("rules") or {}).items():
            if k not in cfg["rules"]:
                raise MonitorError(f"未知规则 {k}")
            if "enabled" in v:
                cfg["rules"][k]["enabled"] = bool(v["enabled"])
            for p, val in (v.get("params") or {}).items():
                if p in cfg["rules"][k]["params"]:
                    cfg["rules"][k]["params"][p] = val
        for k, v in (patch.get("channels") or {}).items():
            if k not in cfg["channels"]:
                raise MonitorError(f"未知渠道 {k}")
            if "enabled" in v:
                cfg["channels"][k]["enabled"] = bool(v["enabled"])
            if v.get("url") is not None and v["url"] != "":       # empty = keep the stored URL
                cfg["channels"][k]["url"] = validate_url(k, v["url"])
            if v.get("clear"):
                cfg["channels"][k]["url"] = cfg["channels"][k]["secret"] = ""
            if v.get("secret"):
                cfg["channels"][k]["secret"] = str(v["secret"]).strip()
        sch = patch.get("schedule") or {}
        if "enabled" in sch:
            cfg["schedule"]["enabled"] = bool(sch["enabled"])
        if "intraday_minutes" in sch:
            cfg["schedule"]["intraday_minutes"] = int(min(60, max(1, sch["intraday_minutes"])))
        if "eod_time" in sch:
            dt.datetime.strptime(sch["eod_time"], "%H:%M")
            cfg["schedule"]["eod_time"] = sch["eod_time"]
        self.save()
        return self.public_config()

    def set_client(self, client: dict) -> None:
        keep = {k: client.get(k) for k in ("holdings", "watch", "pool", "portfolio_settings")}
        keep["holdings"] = [{"code": str(h["code"]), "shares": float(h.get("shares", 0))} for h in keep.get("holdings") or []][:200]
        keep["watch"] = [str(c) for c in keep.get("watch") or []][:200]
        keep["pool"] = [str(c) for c in keep.get("pool") or []][:500]
        keep["updated"] = dt.datetime.now(CN).isoformat(timespec="seconds")
        self.state["client"] = keep
        self.save()

    # ------------------------------------------------------------------ push
    async def push(self, title: str, body: str, level: str = "warn", only: str | None = None) -> dict[str, str]:
        out = {}
        for name in CHANNEL_CN:
            c = self.channel(name)
            if (only and name != only) or (not only and not c["enabled"]) or not c["url"]:
                continue
            try:
                url = validate_url(name, c["url"])
                body_json, params = payload(name, title, body, level, c.get("secret", ""))
                r = await self.http.post(url, json=body_json, params=params or None)
                try:
                    data = r.json()
                except ValueError:
                    data = None
                err = check_response(name, r.status_code, data)
                out[name] = "ok" if err is None else err
            except Exception as exc:  # noqa: BLE001
                out[name] = f"{type(exc).__name__}: {exc}"
        return out

    # ------------------------------------------------------------------ rules
    def _box(self):
        from .assistant.tools import Toolbox
        return Toolbox(self.md, self.pf_getter(), self.state["client"])

    async def rule_vol(self, p: dict) -> list[dict]:
        from .assistant.analysis import analyze_holdings
        h = await analyze_holdings(self._box(), p.get("model", "lw_nonlinear"))
        risk = h.get("risk")
        if not isinstance(risk, dict):
            return []
        vol = risk["ann_vol"]
        if vol <= float(p["threshold"]):
            return []
        top = sorted(risk["risk_contributions"].items(), key=lambda kv: -kv[1]["rc"])[:3]
        body = (f"持仓预期年化波动 **{vol:.1%}**，超过阈值 {float(p['threshold']):.0%}（{risk['model']}，研究数据截至 {risk['asof']}）。\n"
                + "风险贡献最大：" + "，".join(f"{k} {v['rc']:.0%}（权重 {v['w']:.0%}）" for k, v in top))
        return [{"rule": "vol", "key": "vol", "level": "warn", "title": "组合波动超阈值", "body": body}]

    async def rule_drift(self, p: dict) -> list[dict]:
        from .assistant.analysis import analyze_holdings, get_live_portfolio
        box = self._box()
        h = await analyze_holdings(box, "lw_nonlinear")
        t = await get_live_portfolio(box)
        if "error" in h or "error" in t:
            return []
        cur = h["weights_by_market_value"]
        tgt = {x["name"]: x["w"] for x in t["top_holdings"]}
        # top_holdings is truncated: recompute the full target weights through the engine output
        pf = self.pf_getter()
        if pf is not None:
            s = self.state["client"].get("portfolio_settings") or {}
            full = await asyncio.to_thread(pf.optimize, s.get("model") or "pca_factor", "minvar", s.get("max_weight") or 0.1,
                                           s.get("industry_cap"), (self.state["client"].get("pool") if s.get("use_pool") else None) or None)
            tgt = {x["name"]: x["w"] for x in full["weights"]}
        names = set(cur) | set(tgt)
        dev = sorted(((n, cur.get(n, 0.0) - tgt.get(n, 0.0)) for n in names), key=lambda x: -abs(x[1]))
        big = [(n, d) for n, d in dev if abs(d) > float(p["stock"])]
        if not big:
            return []
        body = (f"相对目标组合（{t['label']}）有 **{len(big)}** 只股票偏离超过 {float(p['stock']):.0%}：\n"
                + "\n".join(f"- {n}：当前 {cur.get(n, 0):.1%} / 目标 {tgt.get(n, 0):.1%}（{d:+.1%}）" for n, d in big[:8])
                + "\n可在「持仓调仓」页生成交易单。")
        return [{"rule": "drift", "key": "drift", "level": "info", "title": "持仓偏离目标权重", "body": body}]

    async def regime_vol(self) -> float | None:
        k = await self.md.get_kline("000300.SH", 1)
        closes = np.array([b[4] for b in k.get("bars", [])], dtype=float)
        if len(closes) < 22:
            return None
        return float(np.std(np.diff(np.log(closes[-21:])), ddof=1) * np.sqrt(252))

    async def rule_regime(self, p: dict) -> list[dict]:
        vol = await self.regime_vol()
        if vol is None:
            return []
        state = "high" if vol > float(p["threshold"]) else "low"
        prev = self.state.get("regime_state")
        self.state["regime_state"] = state
        if prev is None or prev == state:
            return []
        cn = {"high": "高波动", "low": "低波动"}
        body = (f"沪深300 近 20 日年化波动 **{vol:.1%}**（阈值 {float(p['threshold']):.1%}），区制由「{cn[prev]}」切换为「{cn[state]}」。\n"
                + ("高波动区制下，最小方差组合相对等权的风险优势通常更明显（见 M0 区制分析）。" if state == "high" else ""))
        return [{"rule": "regime", "key": f"regime:{state}", "level": "warn" if state == "high" else "info", "title": "市场区制切换", "body": body}]

    async def rule_limit(self, p: dict) -> list[dict]:
        cl = self.state["client"]
        held = [h["code"] for h in cl.get("holdings") or []]
        watch = [c for c in (cl.get("watch") or []) if c not in held] if p.get("include_watch", True) else []
        codes = held + watch
        if not codes:
            return []
        q = await self.md.get_quotes(codes)
        opened = session_status()["code"] in ("trading", "break", "closed") and dt.datetime.now(CN).hour >= 9
        out = []
        for c in codes:
            x = q.get(c)
            if not x:
                continue
            kind = "停牌" if opened and x.get("no_trade") else "涨停" if x.get("at_limit_up") else "跌停" if x.get("at_limit_down") else None
            if not kind:
                continue
            where = "持仓" if c in held else "自选"
            effect = {"涨停": "买入受限", "跌停": "卖出受限", "停牌": "无法交易"}[kind]
            out.append({"rule": "limit", "key": f"limit:{c}:{kind}", "level": "warn" if c in held else "info",
                        "title": f"{where}{kind}：{x.get('name', c)}",
                        "body": f"{x.get('name', c)}（{c}）{kind}，现价 {x.get('price')}，涨跌幅 {x.get('pct', 0):+.2%}。{where}股 · {effect}；调仓页会自动冻结相应交易。"})
        return out

    async def rule_data(self, p: dict) -> list[dict]:
        bad = {k: v for k, v in self.md.health.items() if not v.get("ok")}
        return [{"rule": "data", "key": f"data:{k}", "level": "warn", "title": f"数据源异常：{k}",
                 "body": f"{k} 最近一次请求失败（{v.get('at')}）：{v.get('msg') or '未知错误'}。实时页面会回退到备用数据源。"} for k, v in bad.items()]

    # ------------------------------------------------------------------ run
    async def run(self, rules: list[str] | None = None, push: bool = True, force: bool = False) -> list[dict]:
        """Evaluate rules; record and push new events. force=True ignores the once-a-day dedup."""
        async with self.lock:
            cfg = self.state["config"]["rules"]
            names = [r for r in (rules or list(RULES)) if r in RULES and (cfg[r]["enabled"] or rules)]
            today = dt.datetime.now(CN).strftime("%Y-%m-%d")
            fresh: list[dict] = []
            for r in names:
                try:
                    found = await getattr(self, f"rule_{r}")(cfg[r]["params"])
                except Exception as exc:  # noqa: BLE001
                    log.warning("rule %s failed: %s", r, exc)
                    found = [{"rule": r, "key": f"error:{r}", "level": "info", "title": f"规则「{RULES[r]['name']}」检查失败", "body": str(exc)[:300]}]
                for ev in found:
                    k = f"{today}|{ev['key']}"
                    if k in self.state["fired"] and not force:
                        continue
                    self.state["fired"][k] = 1
                    ev.update(id=uuid.uuid4().hex[:10], at=dt.datetime.now(CN).isoformat(timespec="seconds"))
                    ev["pushed"] = await self.push(ev["title"], ev["body"], ev["level"]) if push else {}
                    fresh.append(ev)
            self.state["fired"] = {k: v for k, v in self.state["fired"].items() if k[:10] >= (dt.datetime.now(CN) - dt.timedelta(days=7)).strftime("%Y-%m-%d")}
            self.state["events"] = (fresh[::-1] + self.state["events"])[:300]
            self.last_run = dt.datetime.now(CN).isoformat(timespec="seconds")
            self.save()
            return fresh

    async def tick(self) -> None:
        """One scheduler step (called every 30 s)."""
        sch = self.state["config"]["schedule"]
        if not sch.get("enabled", True):
            return
        now = dt.datetime.now(CN)
        if now.weekday() >= 5:
            return
        sess = session_status()["code"]
        if sess in ("trading", "auction") and time.time() - self.last_intraday >= 60 * int(sch.get("intraday_minutes", 5)):
            self.last_intraday = time.time()
            await self.run(list(INTRADAY))
        eod = dt.datetime.strptime(sch.get("eod_time", "15:10"), "%H:%M").time()
        today = now.strftime("%Y-%m-%d")
        if now.time() >= eod and self.state.get("eod_done") != today:
            self.state["eod_done"] = today
            await self.run()

    async def loop(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:  # noqa: BLE001
                log.exception("monitor tick failed")
            await asyncio.sleep(30)
