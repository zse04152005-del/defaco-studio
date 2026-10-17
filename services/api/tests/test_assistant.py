"""Assistant guardrails and tools (offline: no LLM or network calls)."""
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.assistant.agent import _context_text  # noqa: E402
from app.assistant.guard import check_compliance, check_numbers  # noqa: E402
from app.assistant.tools import FACTOR_CN, FACTOR_HIGH, SCHEMAS, SOURCE_LABEL, Toolbox, call_tool, technicals  # noqa: E402


def test_numbers_traced_with_units_and_rounding():
    ev = [{"total_vol": 0.083111, "price": 1315.87, "amount": 3403630000.0, "sharpe": 0.546526}]
    out = check_numbers("年化波动 8.31%，现价 1315.87 元，成交额 34.04 亿元，Sharpe 0.55，2025-12-31，600519.SH，34 只", ev)
    assert out["total"] == 4 and out["verified"] == 4


def test_invented_numbers_flagged():
    out = check_numbers("波动 7.77%，另一个数 12.5", [{"total_vol": 0.083111}])
    assert out["unverified"] == ["7.77%", "12.5"]


def test_compliance():
    assert check_compliance("建议在 1300 元附近买入") and check_compliance("目标价 1500") and check_compliance("跌破 20 止损")
    assert not check_compliance("模型减配了食品饮料，组合偏防御")


def test_technicals_on_random_walk():
    rng = np.random.default_rng(0)
    c = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    bars = [[f"d{i}", x, x * 1.01, x * 0.99, x, 1e6] for i, x in enumerate(c)]
    t = technicals(bars)
    assert 0 <= t["rsi14"] <= 100 and 0 <= t["pos_in_250d_range"] <= 1 and t["max_drawdown_250d"] <= 0
    assert abs(t["ret_20d"] - (c[-1] / c[-21] - 1)) < 1e-12


def test_schemas_consistent():
    names = {s["name"] for s in SCHEMAS}
    assert names == set(SOURCE_LABEL) and all(hasattr(Toolbox, n) for n in names)
    assert set(FACTOR_CN) == set(FACTOR_HIGH)


def test_offline_tools_and_errors():
    box = Toolbox(md=None)  # tools below do not touch the network
    p = asyncio.run(call_tool(box, "get_portfolio", {"model": "pca_factor"}))
    assert p["n_holdings"] > 0 and abs(sum(p["industry_weights"].values()) - 1) < 0.02
    assert "error" in asyncio.run(call_tool(box, "get_portfolio", {"model": "nope"}))
    assert "error" in asyncio.run(call_tool(box, "no_such_tool", {}))
    s = asyncio.run(call_tool(box, "screen_stocks", {"filters": {"roe": [0.9, 1]}, "top_n": 5}))
    assert s["n_hits"] > 0 and all(x["roe"] >= 0.9 for x in s["top"])


def test_context_never_includes_share_counts():
    txt = _context_text({"page": "rebalance", "holdings": [{"code": "600519.SH", "shares": 1234}],
                         "watch": ["300750.SZ"], "pool": ["601398.SH", "600036.SH"]})
    assert "贵州茅台(600519.SH)" in txt and "1234" not in txt and "元" not in txt
    assert "自选股 1 只" in txt and "候选池 2 只" in txt
