"""Export the trained DeFaCo models (thesis M4, with M3 for comparison) for the demo -> data/models/defaco_m4.json

    .venv/bin/python scripts/export_defaco.py

Reads ``../05_实验结果/models/{m4,m3}/<year>.pkl`` (written by ``03_代码/scripts/run_m4.py`` / ``run_m3.py``)
and keeps only numbers: for every test year the number of factors K, the turnover weight κ chosen on
validation, the characteristic names and the five seeds' Θ = {Γ, α, β} plus the ridge ϱ and the EWMA
half-life. The demo's risk model ``quantkit.riskmodels.defaco`` evaluates these with numpy, so the
demo needs neither torch nor the thesis code. Also stored: the weight of each characteristic in the
loading map, diag(ΓΓ')/tr(ΓΓ') averaged over seeds, for M3 and M4 (the Risk page's Γ panel).

It then writes the DeFaCo portfolio of the demo's rebalancing date to ``web/public/demo-data/portfolio_defaco.json``
(same shape as ``portfolio.json`` ``models[...]``, same constraints as the other precomputed models: 10% cap,
no industry cap), which the pages and the assistant tools merge into the precomputed portfolios when the
live API is down. Both outputs are git-ignored: the trained parameters are research results of the thesis
and are not part of the open-source release. ``scripts/export_demo_data.py`` calls
:func:`add_to_portfolio_json` after rewriting ``portfolio.json``.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("DEFACO_MODELS", ROOT.parent / "05_实验结果" / "models"))
OUT = ROOT / "data" / "models" / "defaco_m4.json"
PORTFOLIO_JSON = ROOT / "web" / "public" / "demo-data" / "portfolio.json"
DEFACO_JSON = ROOT / "web" / "public" / "demo-data" / "portfolio_defaco.json"


def shares(thetas: list[dict]) -> list[float]:
    out = []
    for th in thetas:
        d = (th["gamma"] @ th["gamma"].T).diagonal()
        out.append(d / d.sum())
    return [round(float(x), 6) for x in np.mean(out, axis=0)]


def add_to_portfolio_json(path: Path = PORTFOLIO_JSON, data_dir: Path | None = None) -> bool:
    """Write the precomputed DeFaCo portfolio for the rebalancing date of ``path`` to portfolio_defaco.json."""
    if not (path.exists() and OUT.exists()):
        return False
    sys.path.insert(0, str(ROOT / "services" / "api"))
    os.environ.setdefault("DEFACO_ROOT", str(ROOT))
    from app.portfolio import PortfolioEngine
    from quantkit.config import load_settings

    P = json.loads(path.read_text(encoding="utf-8"))
    if "defaco" in P.get("models", {}):          # older exports kept it inside portfolio.json
        P["models"].pop("defaco")
        path.write_text(json.dumps(P, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    eng = PortfolioEngine.from_store(Path(data_dir) if data_dir else load_settings().data_dir)
    r = eng.optimize("defaco", "minvar", 0.10, None, asof=str(P["asof"])[:10])
    entry = {k: r[k] for k in ("label", "weights", "industry_weights", "risk", "n_holdings", "hhi", "explain")}
    target = path.with_name(DEFACO_JSON.name)
    target.write_text(json.dumps({"asof": P["asof"], "models": {"defaco": entry}}, ensure_ascii=False, separators=(",", ":")),
                      encoding="utf-8")
    print(f"wrote DeFaCo fallback ({r['n_holdings']} holdings, model year {r['explain']['model_year']}) to {target.name}")
    return True


def main() -> int:
    if not (MODELS / "m4").exists():
        print(f"no trained models in {MODELS / 'm4'} (run 03_代码/scripts/run_m4.py first)", file=sys.stderr)
        return 1
    years = {}
    for f in sorted((MODELS / "m4").glob("*.pkl")):
        m4 = pickle.load(open(f, "rb"))
        m3_path = MODELS / "m3" / f.name
        m3 = pickle.load(open(m3_path, "rb")) if m3_path.exists() else None
        years[str(m4["year"])] = {
            "K": int(m4["K"]), "kappa_bp": float(m4["kappa_bp"]), "mu": float(m4["mu"]),
            "char_names": list(m4["char_names"]),
            "seeds": [{"gamma": np.asarray(th["gamma"], dtype=float).tolist(), "alpha": float(th["alpha"]),
                       "beta": float(th["beta"]), "ridge": float(th.get("kappa", 0.01)),
                       "half_life": float(th.get("half_life", 63.0))} for th in m4["thetas"]],
            "share_m4": shares(m4["thetas"]),
            "share_m3": shares(m3["thetas"]) if m3 else None,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"model": "DeFaCo M4 (thesis)", "source": "05_实验结果/models/m4", "years": years},
                              ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB): years {', '.join(years)}")
    add_to_portfolio_json()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
