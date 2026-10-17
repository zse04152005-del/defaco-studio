"""Monitor rules, dedup and webhook payloads (offline: fake quotes, fake webhook server)."""
import asyncio
import base64
import hashlib
import hmac
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.monitor import Monitor, MonitorError, check_response, payload, validate_url  # noqa: E402


class FakeMD:
    def __init__(self, quotes=None, closes=None):
        self.quotes, self.closes, self.health = quotes or {}, closes or [], {}

    async def get_quotes(self, codes):
        return {c: self.quotes[c] for c in codes if c in self.quotes}

    async def get_kline(self, code, years=1):
        return {"bars": [[f"d{i}", c, c, c, c, 1] for i, c in enumerate(self.closes)]}


def make(tmp_path, md, sent):
    def handler(req: httpx.Request):
        sent.append((str(req.url), json.loads(req.content)))
        return httpx.Response(200, json={"errcode": 0, "errmsg": "ok"})
    return Monitor(md, lambda: None, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)), state_file=tmp_path / "m.json")


def test_url_whitelist():
    assert validate_url("wecom", "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=x")
    for ch, bad in [("wecom", "http://qyapi.weixin.qq.com/x"), ("feishu", "https://evil.example.com/open.feishu.cn"),
                    ("dingtalk", "https://127.0.0.1/robot")]:
        with pytest.raises(MonitorError):
            validate_url(ch, bad)


def test_payloads_and_signatures():
    w, _ = payload("wecom", "T", "B")
    assert w["msgtype"] == "markdown" and "T" in w["markdown"]["content"]
    f, _ = payload("feishu", "T", "B", secret="sec")
    ts = f["timestamp"]
    assert f["sign"] == base64.b64encode(hmac.new(f"{ts}\nsec".encode(), b"", hashlib.sha256).digest()).decode()
    d, params = payload("dingtalk", "T", "B", secret="sec")
    assert d["msgtype"] == "markdown" and set(params) == {"timestamp", "sign"}
    assert check_response("wecom", 200, {"errcode": 93000, "errmsg": "invalid webhook url"})
    assert check_response("feishu", 200, {"code": 0}) is None


def test_limit_rule_push_and_dedup(tmp_path):
    md = FakeMD(quotes={"600519.SH": {"name": "贵州茅台", "price": 10.0, "pct": 0.1, "at_limit_up": True},
                        "000001.SZ": {"name": "平安银行", "price": 9.0, "pct": -0.001}})
    sent = []
    m = make(tmp_path, md, sent)
    m.update_config({"channels": {"wecom": {"enabled": True, "url": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=k"}}})
    m.set_client({"holdings": [{"code": "600519.SH", "shares": 100}], "watch": ["000001.SZ"]})
    ev = asyncio.run(m.run(["limit"]))
    assert len(ev) == 1 and "涨停" in ev[0]["title"] and ev[0]["pushed"] == {"wecom": "ok"}
    assert len(sent) == 1 and sent[0][0].startswith("https://qyapi.weixin.qq.com/")
    assert asyncio.run(m.run(["limit"])) == []            # same event, same day: not again
    assert len(asyncio.run(m.run(["limit"], force=True))) == 1
    saved = json.loads((tmp_path / "m.json").read_text())
    assert saved["client"]["holdings"][0]["shares"] == 100 and len(saved["events"]) == 2
    assert m.public_config()["channels"]["wecom"]["url_masked"] != "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=k"


def test_regime_fires_only_on_change(tmp_path):
    calm = [100 * (1.001 ** i) for i in range(40)]
    md = FakeMD(closes=calm)
    m = make(tmp_path, md, [])
    assert asyncio.run(m.run(["regime"], push=False)) == []          # first observation: record the state only
    md.closes = calm + [calm[-1] * f for f in (1.05, 0.95, 1.06, 0.94, 1.05, 0.95)]
    ev = asyncio.run(m.run(["regime"], push=False))
    assert len(ev) == 1 and "高波动" in ev[0]["body"]
    assert asyncio.run(m.run(["regime"], push=False)) == []


def test_disabled_channel_not_pushed_and_bad_url_rejected(tmp_path):
    sent = []
    m = make(tmp_path, FakeMD(), sent)
    with pytest.raises(MonitorError):
        m.update_config({"channels": {"feishu": {"url": "https://example.com/hook"}}})
    m.update_config({"channels": {"feishu": {"enabled": False, "url": "https://open.feishu.cn/open-apis/bot/v2/hook/x"}}})
    assert asyncio.run(m.push("t", "b")) == {} and sent == []
    assert asyncio.run(m.push("t", "b", only="feishu")) == {"feishu": "ok"}
