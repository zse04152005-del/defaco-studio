"""Synthetic A-share market with known factor structure, used by all tests (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantkit.data.panel import MarketPanel

N_STOCKS, N_DAYS, K = 40, 700, 3


def make_codes(n: int) -> list[str]:
    codes = []
    for i in range(n):
        if i % 4 == 0:
            codes.append(f"{600000 + i:06d}.SH")
        elif i % 4 == 1:
            codes.append(f"{1 + i:06d}.SZ")
        elif i % 4 == 2:
            codes.append(f"{300000 + i:06d}.SZ")
        else:
            codes.append(f"{601000 + i:06d}.SH")
    return codes


@pytest.fixture(scope="session")
def synth():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2019-01-01", periods=N_DAYS)
    codes = make_codes(N_STOCKS)
    groups = np.array([i % 4 for i in range(N_STOCKS)])
    B = rng.normal(0, 1, (N_STOCKS, K)) * [1.0, 0.5, 0.3] + [1.0, 0, 0]
    f = rng.normal(0, 0.008, (N_DAYS, K))
    ind = rng.normal(0, 0.004, (N_DAYS, 4))
    eps = rng.normal(0, 0.012, (N_DAYS, N_STOCKS))
    ret = f @ B.T + ind[:, groups] + eps
    ret = np.clip(ret, -0.095, 0.095)
    R = pd.DataFrame(ret, index=dates, columns=codes)

    # stock 5 suspended for 30 days; stock 7 lists late (day 500); stock 9 is ST at the end
    R.iloc[300:330, 5] = np.nan
    R.iloc[:500, 7] = np.nan
    price = 10 * (1 + R.fillna(0)).cumprod()
    price.iloc[:500, 7] = np.nan

    rows = []
    for c_i, c in enumerate(codes):
        p = price[c]
        pre = p.shift(1)
        for d_i, d in enumerate(dates):
            if np.isnan(p.iloc[d_i]):
                continue
            trading = not np.isnan(R.iloc[d_i, c_i])
            r = R.iloc[d_i, c_i] if trading else np.nan
            rows.append({"date": d, "code": c, "close": round(float(p.iloc[d_i]), 2),
                         "preclose": float(pre.iloc[d_i]) if not np.isnan(pre.iloc[d_i]) else np.nan,
                         "open": float(p.iloc[d_i]), "high": float(p.iloc[d_i]), "low": float(p.iloc[d_i]),
                         "volume": 1e6 if trading else 0.0, "amount": 1e7 * (1 + c_i) if trading else 0.0,
                         "turn": 1.0 + 0.1 * c_i if trading else 0.0, "tradestatus": int(trading),
                         "isST": int(c_i == 9 and d_i > 600), "ret": r,
                         "peTTM": 10 + c_i, "pbMRQ": 1 + 0.1 * c_i, "psTTM": 2.0, "pcfNcfTTM": 5.0})
    bars = pd.DataFrame(rows)
    # force a limit-up close for stock 3 on day 650 (10% board)
    d650 = dates[650]
    m = (bars["date"] == d650) & (bars["code"] == codes[3])
    pre650 = bars.loc[m, "preclose"].iloc[0]
    bars.loc[m, "close"] = round(pre650 * 1.10, 2)

    members = pd.DataFrame({"snapshot": dates[0], "code": codes, "index": "hs300"})
    industry = pd.DataFrame({"code": codes, "start": dates[0], "sw_l1": [f"I{g}" for g in groups]})
    market = R.mean(axis=1)
    panel = MarketPanel.from_long(bars, calendar=dates, members=members, industry=industry, market_ret=market)
    return {"panel": panel, "B": B, "groups": groups, "codes": codes, "dates": dates, "R": R, "bars": bars}
