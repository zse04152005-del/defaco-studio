# DeFaCo Studio — web

React 19 + Vite + Tailwind 4. Charts: TradingView Lightweight Charts (K-line) and ECharts.
Animated controls are adapted from Uiverse.io (MIT) in `src/styles/uiverse.css`.

```bash
npm install
../scripts/dev.sh      # API on :8000 + web on :5173 (Vite proxies /api and /ws)
```

Real-time layer: `src/lib/realtime.ts` keeps one shared WebSocket (auto-reconnect, REST polling
fallback). Live data appears in the top bar, overview (heatmap, breadth, index cards), stock page
(price flash, 分时 chart, 五档盘口, daily K up to today) and rebalance page (live prices).

Data: until the FastAPI service exists, pages read static JSON from `public/demo-data/`,
produced by `DEFACO_DATA=data_smoke ../.venv/bin/python ../scripts/export_demo_data.py`.
Every file carries a `meta` block; smoke data (survivorship-biased) shows a warning banner.

Visual check (renders every page in local Chrome, saves screenshots, reports console errors):

```bash
node scripts/snap.mjs /tmp/shots "overview:/,stock:/stock?code=600519.SH,portfolio:/portfolio" dark
```

| Page | Route | Paper module shown |
|---|---|---|
| 市场总览 | `/` | regime monitor (Module R) |
| 个股 K 线 | `/stock` | characteristic profile (CIL inputs) |
| 选股器 | `/screener` | characteristic percentiles |
| 组合构建 | `/portfolio` | risk models + long-only min-variance (DPL) |
| 风险解析 | `/risk` | factor / industry-block / idiosyncratic split (CIL, SIC), Γ heatmap placeholder |
| 持仓调仓 | `/rebalance` | lot-size trade list, costs |
| 回测绩效 | `/backtest` | net-of-cost backtests |
| 监控预警 | `/alerts` | rules & channels (UI only) |
| 论文实验室 | `/lab` | method figures, ablation ladder, 2×2 turnover experiment |
