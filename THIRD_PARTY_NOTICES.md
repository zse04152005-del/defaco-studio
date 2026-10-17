# Third-party notices

| Component | Use | License |
|---|---|---|
| numpy, pandas, scipy, scikit-learn, pyarrow, duckdb, cvxpy, pyyaml, pydantic | runtime dependencies | BSD / MIT / Apache-2.0 |
| akshare | data access (dependency) | MIT |
| baostock | data access (dependency) | see package |
| Analytical nonlinear shrinkage (`quantkit/riskmodels/baselines.py`) | independent Python port of the algorithm published by Ledoit & Wolf (2020, *Annals of Statistics* 48(5)); reference implementation also available at https://github.com/pald22/covShrinkage | algorithm from the paper; covShrinkage is MIT |

DeFaCo Studio itself is released under the MIT License (see `LICENSE`). Code copied or adapted from other
projects must be listed here with its original copyright and license, and keep the original header in the copied file.

## Web demo (`web/`)

| Component | Use | License |
|---|---|---|
| Uiverse.io elements (https://github.com/uiverse-io/galaxy) | adapted CSS for buttons, switches, loaders, search input, segmented control in `web/src/styles/uiverse.css` — each block names its original author/element | MIT, Copyright (c) 2023 Uiverse.io |
| TradingView Lightweight Charts™ | K-line charts; the attribution logo is kept and the K-line panel credits TradingView | Apache-2.0, Copyright (c) TradingView, Inc. |
| Apache ECharts, echarts-for-react | analytical charts | Apache-2.0 / MIT |
| React, React Router, Zustand, Vite, Tailwind CSS, clsx, lucide-react | UI framework and tooling | MIT / ISC |
| playwright-core (dev only) | headless visual checks (`web/scripts/snap.mjs`) | Apache-2.0 |

## Research & API services

| Component | Use | License |
|---|---|---|
| arch (Kevin Sheppard) | GARCH(1,1) estimation in `quantkit.dynamics` | NCSA-style permissive |
| PyTorch (optional) | autoencoder factor extractors | BSD-3-Clause |
| matplotlib | backtest report figures | Matplotlib (PSF-based) |
| FastAPI, uvicorn, httpx | demo API service | MIT / BSD-3-Clause / BSD-3-Clause |
| pytdx | optional TDX intraday source in `services/api` (not vendored; the project declares no license, so it is only an optional runtime dependency for personal research) | none declared |
| Tencent / Sina public quote endpoints | demo real-time display only (may be delayed; personal research use) | terms of the providers |

## AI assistant (v0.3)

| Component | License | Use |
|---|---|---|
| react-markdown 9 | MIT | Rendering assistant answers |
| remark-gfm 4 | MIT | GitHub-flavoured Markdown tables in answers |
| DeepSeek API | Commercial API (user-supplied key) | LLM; called only from the local API server, key kept in `.env` |

The assistant workflow takes ideas, but no code, from TradingAgents (Apache-2.0; bull/bear/risk roles),
OpenBB Copilot and FinRobot (tool-grounded answers with citations) and dexter (no license; ideas only).

## Securities search (v0.3.1)

| Component | License | Use |
|---|---|---|
| pypinyin 0.55 | MIT | Pinyin initials for the securities search (build time only, `scripts/export_securities.py`) |
| BaoStock `query_stock_basic` / AkShare `stock_info_bj_name_code` | free public data | Full A-share list (Shanghai/Shenzhen; Beijing) |
| Sina `CN_MarketData.getKLineData` | free public endpoint | Daily K-line fallback when Tencent has no history (e.g. BSE 920xxx codes); unadjusted, labelled in the UI |

## Factor library, factor research and backtesting (v0.4)

| Component | License | Use |
|---|---|---|
| Qlib Alpha158 feature expressions (microsoft/qlib, `qlib/contrib/data/loader.py`) | MIT | Re-implemented formulas in `quantkit/features/alphas.py` (no code copied) |
| WorldQuant "101 Formulaic Alphas" (Kakushadze, 2016, Wilmott) | published paper | Formulas re-implemented; level-dependent ones in relative form (flagged) |
| Guotai Junan "191 short-horizon alphas" research report (2017) | published report | Formulas re-implemented (subset) |
| alphalens (Quantopian, now alphalens-reloaded) | Apache-2.0 | Methodology only (IC, quantile returns, turnover, decay); own implementation in `quantkit/factor_eval.py` |
| Qlib backtest `Exchange` design | MIT | Execution-model ideas (limit thresholds, min cost, impact cost, failed orders stay in cash); own implementation |
| bottleneck 1.6 | BSD-2-Clause | Fast moving-window operators |
| cvxpy (Clarabel solver) | Apache-2.0 | Risk parity / mean-variance programmes |
| scipy (hierarchical clustering) | BSD-3-Clause | HRP |
