import numpy as np
import pandas as pd

from quantkit.factor_eval import FactorLab


def make_lab(T=40, N=200, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2015-01-31", periods=T, freq="ME")
    codes = pd.Index([f"{600000 + i}.SH" for i in range(N)])
    ind = np.array([f"I{i % 5}" for i in range(N)])
    fwd = rng.normal(0, 0.05, (T, N)) + (np.arange(N) % 5)[None, :] * 0.01        # industry effect in returns
    rows = []
    for t, d in enumerate(months):
        rows.append(pd.DataFrame({"date": d, "code": codes, "in300": True, "industry": ind,
                                  "log_mcap": rng.normal(10, 1, N), "fwd1": fwd[t], "fwd3": fwd[t], "fwd6": fwd[t]}))
    base = pd.concat(rows, ignore_index=True)
    values = {"perfect": fwd.astype(np.float32), "noise": rng.normal(0, 1, (T, N)).astype(np.float32),
              "industry": np.tile((np.arange(N) % 5).astype(np.float32), (T, 1))}
    return FactorLab(months=months, codes=codes, values=values, base=base)


def test_ic_signs_and_magnitudes():
    lab = make_lab()
    s = lab.summary(universe="csi800", neutral="none", h=1)
    assert s.loc["perfect", "rank_ic"] > 0.99 and abs(s.loc["noise", "rank_ic"]) < 0.05
    assert s.loc["industry", "rank_ic"] > 0.1                       # industry drives returns before neutralising
    assert abs(s.loc["perfect", "monotonic"] - 1.0) < 1e-9 and s.loc["perfect", "ls_ann"] > 0


def test_industry_neutralisation_removes_industry_factor():
    lab = make_lab()
    P = lab.prepared("csi800", "industry")
    j = P["names"].index("industry")
    assert all(np.nanmax(np.abs(np.nan_to_num(Z[:, j]))) < 1e-6 or np.isnan(Z[:, j]).all() for Z in P["Z"])


def test_detail_shapes():
    lab = make_lab()
    d = lab.detail("perfect", "csi800", "none", 1)
    assert len(d["ic_series"]) == 40 and len(d["decay"]) == 6 and d["decay"][0] > 0.99
    assert set(d["quantile_mean"]) == {1, 2, 3, 4, 5} and d["quantile_mean"][5] > d["quantile_mean"][1]
