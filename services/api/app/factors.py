"""Factor research (F4): the factor library (24 characteristics + Alpha158 / Alpha101 / GTJA191)
evaluated alphalens-style on the research data. See quantkit.factor_eval for the method.

    GET  /api/factors/summary?universe=csi800|csi300&neutral=none|industry|industry_size&h=1|3|6
    GET  /api/factors/detail/{name}?...
    POST /api/factors/corr {names, universe, neutral}
"""
from __future__ import annotations

import math
import threading
from pathlib import Path

from quantkit.data.store import DataStore
from quantkit.factor_eval import FactorLab, catalogue

UNIVERSES = {"csi800": "沪深300 + 中证500 历史成分股", "csi300": "沪深300 历史成分股"}
NEUTRALS = {"none": "不中性化", "industry": "行业中性", "industry_size": "行业 + 市值中性"}


def clean(x):
    if isinstance(x, float):
        return None if not math.isfinite(x) else round(x, 6)
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, "item"):
        return clean(x.item())
    return x


class FactorService:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._lab: FactorLab | None = None
        self._lock = threading.Lock()
        self._summary: dict = {}
        self._detail: dict = {}

    def lab(self) -> FactorLab:
        with self._lock:
            if self._lab is None:
                self._lab = FactorLab.load(DataStore(self.data_dir))
            return self._lab

    def check(self, universe: str, neutral: str, h: int) -> None:
        if universe not in UNIVERSES or neutral not in NEUTRALS or h not in (1, 3, 6):
            raise ValueError("参数错误：universe ∈ csi800/csi300，neutral ∈ none/industry/industry_size，h ∈ 1/3/6")

    def summary(self, universe="csi800", neutral="industry_size", h=1) -> dict:
        self.check(universe, neutral, h)
        key = (universe, neutral, h)
        if key not in self._summary:
            lab = self.lab()
            s = lab.summary(universe, neutral, h)
            cat = catalogue()
            rows = []
            for n, r in s.iterrows():
                c = cat.get(n)
                rows.append({"name": n, "formula": c.formula if c else "", "desc": c.desc if c else "", "modified": bool(c.modified) if c else False,
                             **{k: (v if not hasattr(v, "item") else v.item()) for k, v in r.items()}})
            self._summary[key] = clean({"universe": UNIVERSES[universe], "neutral": NEUTRALS[neutral], "h": h,
                                        "start": str(lab.months[0].date()), "end": str(lab.months[-1].date()),
                                        "n_factors": len(rows), "rows": rows})
        return self._summary[key]

    def detail(self, name: str, universe="csi800", neutral="industry_size", h=1) -> dict:
        self.check(universe, neutral, h)
        lab = self.lab()
        if name not in lab.values:
            raise KeyError(name)
        key = (name, universe, neutral, h)
        if key not in self._detail:
            self._detail[key] = clean(lab.detail(name, universe, neutral, h))
            if len(self._detail) > 200:
                self._detail.pop(next(iter(self._detail)))
        return self._detail[key]

    def corr(self, names: list[str], universe="csi800", neutral="industry_size") -> dict:
        lab = self.lab()
        names = [n for n in names if n in lab.values][:30]
        C = lab.corr(names, universe, neutral)
        return clean({"names": names, "matrix": C.round(3).to_numpy().tolist()})
