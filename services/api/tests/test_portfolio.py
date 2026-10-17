"""Live portfolio optimisation (offline: synthetic panel; one optional check on the research data)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantkit.data.panel import MarketPanel  # noqa: E402

from app import main  # noqa: E402
from app.portfolio import MODELS, OptimizeError, PortfolioEngine  # noqa: E402

N, T = 30, 640
INDUSTRIES = ["48", "34", "41", "27", "63"]          # 银行 食品饮料 公用事业 电子 电力设备
REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def engine() -> PortfolioEngine:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=T)
    codes = [f"{600000 + i:06d}.SH" for i in range(N)]
    groups = np.arange(N) % len(INDUSTRIES)
    # one market factor + an industry factor + noise: low-vol industries attract the min-variance weight
    vol = np.array([0.006, 0.012, 0.007, 0.02, 0.018])[groups]
    ret = rng.normal(0, 0.009, (T, 1)) * rng.uniform(0.5, 1.5, N) + rng.normal(0, 0.006, (T, 5))[:, groups] \
        + rng.normal(0, 1, (T, N)) * vol
    close = 10 * np.exp(np.cumsum(ret, axis=0))
    bars = pd.DataFrame({
        "date": np.repeat(dates, N), "code": np.tile(codes, T), "ret": ret.ravel(), "close": close.ravel(),
        "preclose": np.vstack([close[:1], close[:-1]]).ravel(), "amount": 1e8, "turn": 1.0,
        "volume": np.tile(rng.uniform(1e6, 1e7, N), T), "tradestatus": 1, "isST": 0})
    bars.loc[bars["code"] == codes[-1], "isST"] = 1                       # ST -> not investable
    bars.loc[(bars["code"] == codes[-2]) & (bars["date"] == dates[-1]), "tradestatus"] = 0  # suspended on asof
    members = pd.DataFrame({"snapshot": dates[-1], "code": codes, "name": [f"股票{i}" for i in range(N)], "index": "hs300"})
    industry = pd.DataFrame({"code": codes, "start": dates[0], "sw_l1": [INDUSTRIES[g] for g in groups]})
    panel = MarketPanel.from_long(bars, members=members, industry=industry,
                                  market_ret=pd.Series(ret.mean(axis=1), index=dates))
    return PortfolioEngine(panel=panel, names=dict(zip(members["code"], members["name"])), dataset="synthetic")


def _check(out: dict, max_weight: float, industry_cap: float | None) -> None:
    w = pd.Series({x["code"]: x["w"] for x in out["weights"]})
    assert abs(w.sum() - 1) < 1e-3                       # only holdings >= 1bp are listed
    assert w.min() > 0 and w.max() <= max_weight + 1e-6
    if industry_cap is not None:
        assert max(out["industry_weights"].values()) <= industry_cap + 1e-6
    assert abs(sum(x["rc"] for x in out["weights"]) - 1) < 1e-3
    assert out["n_holdings"] == len(w) and abs(out["hhi"] - (w ** 2).sum()) < 1e-5
    assert out["risk"]["total_vol"] > 0


@pytest.mark.parametrize("model", list(MODELS))
def test_every_model_respects_caps(engine, model):
    if model == "defaco" and not (REPO / "data" / "models" / "defaco_m4.json").exists():
        pytest.skip("DeFaCo models not exported (scripts/export_defaco.py)")
    out = engine.optimize(model=model, max_weight=0.08, industry_cap=0.30)
    _check(out, 0.08, 0.30)
    assert out["universe_size"] == N - 2 and out["label"] == MODELS[model]
    if model == "pca_factor":
        assert abs(out["risk"]["factor_share"] + out["risk"]["idio_share"] - 1) < 1e-9
        assert len(out["risk"]["exposures"]) > 0
    elif model == "defaco":   # factor + industry-block + specific = total
        r = out["risk"]
        assert abs(r["factor_share"] + r["industry_share"] + r["idio_share"] - 1) < 1e-5   # shares are rounded to 6 decimals
        assert out["explain"]["model_year"] in range(2016, 2026)
    else:
        assert "factor_share" not in out["risk"]


def test_caps_change_the_portfolio(engine):
    loose = engine.optimize(model="lw_nonlinear", max_weight=0.20, industry_cap=None)
    tight = engine.optimize(model="lw_nonlinear", max_weight=0.06, industry_cap=0.25)
    _check(loose, 0.20, None)
    _check(tight, 0.06, 0.25)
    assert max(loose["industry_weights"].values()) > 0.25          # the industry cap binds
    assert tight["n_holdings"] > loose["n_holdings"]
    assert tight["risk"]["total_vol"] >= loose["risk"]["total_vol"] - 1e-9


def test_custom_universe_drops_untradable(engine):
    codes = list(engine.panel.codes)
    pool = codes[:10] + [codes[-1], codes[-2], "999999.SH"]
    out = engine.optimize(model="pca_factor", max_weight=0.25, industry_cap=0.5, universe=pool)
    _check(out, 0.25, 0.5)
    assert out["universe_source"] == "custom" and out["universe_size"] == 10
    assert set(out["dropped"]) == {codes[-1], codes[-2], "999999.SH"}
    assert {x["code"] for x in out["weights"]} <= set(codes[:10])
    assert len(out["corr"]["codes"]) == out["n_holdings"] == len(out["corr"]["matrix"])


def test_infeasible(engine):
    codes = list(engine.panel.codes)
    with pytest.raises(OptimizeError, match="不可行"):
        engine.optimize(universe=codes[:5], max_weight=0.10)
    with pytest.raises(OptimizeError, match="行业"):
        engine.optimize(max_weight=0.10, industry_cap=0.15)          # 5 industries x 15% < 100%
    with pytest.raises(OptimizeError, match="未知目标"):
        engine.optimize(objective="nope")


@pytest.mark.parametrize("objective", ["rp", "hrp", "mv"])
def test_other_objectives(engine, objective):
    r = engine.optimize(objective=objective, max_weight=0.2, industry_cap=None)
    w = [x["w"] for x in r["weights"]]
    assert abs(sum(w) - 1) < 1e-4 and max(w) <= 0.2 + 1e-4
    if objective in ("rp", "hrp"):
        assert r["n_holdings"] == r["universe_size"]                  # risk parity / HRP hold every name
    if objective == "mv":
        assert 0 <= r["expected_returns"]["shrinkage"] <= 1


def test_endpoint(engine, monkeypatch):
    monkeypatch.setattr(main, "pf", engine)
    client = TestClient(main.app)                                    # no lifespan: no data load, no network
    r = client.post("/api/portfolio/optimize", json={"model": "ewma", "max_weight": 0.1, "industry_cap": 0.3})
    assert r.status_code == 200
    _check(r.json(), 0.1, 0.3)
    assert client.post("/api/portfolio/optimize", json={"objective": "hrp", "max_weight": 0.2}).status_code == 200
    assert client.post("/api/portfolio/optimize", json={"model": "nope"}).status_code == 422
    assert client.post("/api/portfolio/optimize", json={"max_weight": 1.5}).status_code == 422
    assert client.get("/api/portfolio/options").json()["ready"] is True


@pytest.mark.skipif(not (REPO / "data" / "curated" / "bars.parquet").exists(), reason="research data not present")
def test_research_data_matches_precomputed_json():
    eng = PortfolioEngine.from_store(REPO / "data")
    out = eng.optimize(model="pca_factor", max_weight=0.10, industry_cap=None, asof="2025-12-31")
    ref = pd.read_json(REPO / "web" / "public" / "demo-data" / "portfolio.json", typ="series")["models"]["pca_factor"]
    assert out["universe_size"] == 100 and out["n_holdings"] == ref["n_holdings"]
    assert abs(out["risk"]["total_vol"] - ref["risk"]["total_vol"]) < 1e-3
    small = eng.optimize(model="lw_linear", max_weight=0.2, industry_cap=0.4,
                         universe=["600519.SH", "000858.SZ", "600036.SH", "601318.SH", "600900.SH", "601398.SH", "000333.SZ"])
    _check(small, 0.2, 0.4)
