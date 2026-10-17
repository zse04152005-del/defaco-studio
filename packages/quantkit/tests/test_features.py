import numpy as np
import pandas as pd

from quantkit.features.characteristics import ALL_CHARS, compute_characteristics, fundamental_characteristics
from quantkit.features.transforms import characteristic_matrix, rank_normalize


def month_ends(dates):
    s = pd.Series(dates)
    return pd.DatetimeIndex(s.groupby(s.dt.to_period("M")).max().values)


def test_compute_characteristics_shapes_and_ranges(synth):
    p = synth["panel"]
    me = month_ends(synth["dates"])[12:]
    long = compute_characteristics(p, me)
    assert list(long.columns[2:]) == ALL_CHARS
    last = long[long["date"] == me[-1]].set_index("code")
    assert last["size"].notna().sum() >= 35
    assert last["mom_12_2"].notna().sum() >= 35
    assert (last["high52"].dropna() <= 1 + 1e-12).all()
    # betas of the synthetic market should average roughly 1
    assert 0.6 < last["beta_12m"].mean() < 1.4


def test_characteristics_do_not_use_future_prices(synth):
    p = synth["panel"]
    me = month_ends(synth["dates"])
    t = me[20]
    base = compute_characteristics(p, me[[20]]).set_index("code")
    # corrupt every return after t: characteristics at t must not change
    p2 = type(p)(**{**p.__dict__})
    p2.ret = p.ret.copy()
    p2.ret.loc[p2.ret.index > t] = 0.5
    again = compute_characteristics(p2, me[[20]]).set_index("code")
    cols = ["mom_12_2", "rev_1m", "vol_1m", "beta_12m", "ivol_3m", "high52", "maxret_1m"]
    pd.testing.assert_frame_equal(base[cols], again[cols])


def test_fundamentals_are_point_in_time():
    me = pd.DatetimeIndex(["2020-03-31", "2020-04-30", "2020-05-29"])
    codes = pd.Index(["600000.SH"])
    yjbb = pd.DataFrame({
        "code": ["600000.SH", "600000.SH"],
        "period": pd.to_datetime(["2019-12-31", "2020-03-31"]),
        "ann_date": pd.to_datetime(["2020-04-20", "2020-04-28"]),
        "roe": [10.0, 3.0], "gross_margin": [30.0, 31.0], "revenue_yoy": [5.0, 6.0], "profit_yoy": [1.0, 2.0],
    })
    out = fundamental_characteristics(me, codes, yjbb, None)["roe"]["600000.SH"]
    assert np.isnan(out.iloc[0])            # nothing announced by March 31
    assert out.iloc[1] == 3.0               # Q1 report (announced Apr 28) is the latest by Apr 30
    assert out.iloc[2] == 3.0


def test_rank_normalize_range_and_missing():
    df = pd.DataFrame({"date": ["d"] * 5, "code": list("abcde"), "x": [1.0, 2.0, np.nan, 4.0, 5.0]})
    out = rank_normalize(df)
    assert out["x"].between(-0.5, 0.5).all()
    assert out.loc[2, "x"] == 0.0
    z = characteristic_matrix(out.assign(date=pd.Timestamp("2020-01-31")), pd.Timestamp("2020-01-31"),
                              ["a", "zz"], ["x"])
    assert list(z.columns) == ["const", "x"] and z.loc["zz", "x"] == 0.0


def test_ttm_eps():
    from quantkit.pipeline import _ttm
    fin = pd.DataFrame({
        "code": ["X"] * 3,
        "period": pd.to_datetime(["2019-03-31", "2019-12-31", "2020-03-31"]),
        "ann_date": pd.to_datetime(["2019-04-25", "2020-03-30", "2020-04-28"]),
        "eps": [0.4, 2.0, 0.5],
    })
    out = _ttm(fin, "eps").set_index("period")["eps_ttm"]
    assert out[pd.Timestamp("2019-12-31")] == 2.0
    assert abs(out[pd.Timestamp("2020-03-31")] - 2.1) < 1e-12
    assert np.isnan(out[pd.Timestamp("2019-03-31")])   # previous-year inputs unavailable


def test_high52_missing_before_listing(synth):
    p = synth["panel"]
    me = month_ends(synth["dates"])
    long = compute_characteristics(p, me).set_index(["date", "code"])
    late = synth["codes"][7]                     # lists on day 500
    early_month = me[me < synth["dates"][500]][-1]
    assert np.isnan(long.loc[(early_month, late), "high52"])


def test_point_in_time_dates():
    from quantkit.data.ingest import legal_deadline, point_in_time_dates
    per = pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"])
    assert list(legal_deadline(pd.Series(per)).dt.strftime("%Y-%m-%d")) == \
        ["2023-04-30", "2023-08-31", "2023-10-31", "2024-04-30"]
    fin = pd.DataFrame({"code": ["A", "A", "B"], "period": pd.to_datetime(["2023-12-31", "2024-03-31", "2023-12-31"]),
                        "ann_date": pd.to_datetime(["2025-04-20", "2024-04-15", "2025-04-01"]), "eps": [1, 2, 3]})
    disc = pd.DataFrame({"code": ["A"], "period": pd.to_datetime(["2023-12-31"]), "disclosed": pd.to_datetime(["2024-03-28"])})
    out = point_in_time_dates(fin, disc).set_index(["code", "period"])["ann_date"]
    assert out[("A", pd.Timestamp("2023-12-31"))] == pd.Timestamp("2024-03-28")   # actual disclosure
    assert out[("A", pd.Timestamp("2024-03-31"))] == pd.Timestamp("2024-04-15")   # em date < deadline
    assert out[("B", pd.Timestamp("2023-12-31"))] == pd.Timestamp("2024-04-30")   # capped at deadline
