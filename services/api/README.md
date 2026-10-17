# DeFaCo Studio API (v0.1)

Real-time market data for the demo — **display layer only**; portfolio maths uses official closes.

```bash
.venv/bin/uvicorn app.main:app --app-dir services/api --port 8000   # or scripts/dev.sh
```

| Endpoint | Source (fallback order) |
|---|---|
| `GET /api/market/status` | session clock (Beijing time) + source health |
| `GET /api/realtime/quotes?codes=600519.SH,...` | Tencent → Sina (batched, 2 s cache) |
| `GET /api/realtime/minute/{code}` | Tencent → pytdx (if a TDX server is reachable) → AkShare/Sina |
| `GET /api/kline/{code}?years=3` | Tencent forward-adjusted daily bars (≤640 per call, paged) |
| `WS /ws/quotes` | push every 3 s while trading, 30 s otherwise; immediate reply on subscribe |

## 组合优化（F5/F6）

`POST /api/portfolio/optimize` — 实时最小方差组合，返回与 `web/public/demo-data/portfolio.json`
`models[...]` 相同的结构（另含 `asof`、`universe_size`、`dropped`、`constraints`、`corr`、`timing_ms`）。

```json
{"model": "pca_factor", "objective": "minvar", "max_weight": 0.10, "industry_cap": 0.35,
 "universe": null, "asof": null}
```

- `model`：`sample` / `lw_linear` / `lw_nonlinear` / `ewma` / `pca_factor`
- `objective`：只实现了 `minvar`；`rp` / `hrp` / `mv` 返回 501
- `universe`：股票代码列表（选股器候选池）；为空时 = 沪深300 中流通市值前 100
- 约束不可行（例如 5 只股票 × 10% 上限）返回 422，`detail` 为中文说明
- 数据：`$DEFACO_DATA`（默认 `data`）。启动时在后台加载最近 3 年行情（约 4 秒），之后单次请求约 50–120 ms
- `GET /api/portfolio/options`：可选模型 / 目标，以及面板是否已加载

Free public endpoints: may be delayed or change without notice; for personal research and
teaching. Public deployment of real-time exchange data requires a data licence — show delayed
or end-of-day data instead. Interactive docs: http://127.0.0.1:8000/docs

## AI 投研助手（DeepSeek）

1. 复制 `.env.example` 为仓库根目录的 `.env`，填入 `DEEPSEEK_API_KEY`（`.env` 已被 git 忽略）。
2. 重启 API 后，`GET /api/assistant/status` 应返回 `"ready": true`。状态接口不会返回 Key。

| 接口 | 作用 |
|---|---|
| `POST /api/assistant/chat` `{messages, context}` | SSE 流式问答（工具调用智能体） |
| `POST /api/assistant/report` `{code, context}` | SSE 流式个股研报（数据 → 多空分析师 → 风控主编） |

SSE 事件类型：
- `status`、`reasoning`、`delta`
- `round_end`：本轮文字是规划说明
- `tool_start` / `tool_end`、`stage`
- `final`：来源、数字核验、合规提醒、token 用量
- `error`

配置项：
- 模型：`DEEPSEEK_MODEL`（默认 `deepseek-flash`）、`DEEPSEEK_MODEL_PRO`（研报主编，默认 `deepseek-v4-pro`）
- `ASSISTANT_ENABLED=0` 关闭助手

测试：`.venv/bin/python -m pytest services/api/tests`（离线，不调用大模型）。

## K 线数据源

`GET /api/kline/{code}` 默认使用腾讯前复权日线。腾讯返回少于 30 根时（例如北交所 920 开头的新代码只有当天一根），回退到新浪日线：不复权，`source = "sina-unadjusted"`，前端会注明。

## 证券列表（搜索用）

```bash
python scripts/ingest.py --steps basic      # BaoStock 全部证券 + 北交所列表 -> data/raw/stock_basic*.parquet
python scripts/export_securities.py         # -> web/public/demo-data/securities_all.json（含拼音首字母）
```

## 监控与推送

| 接口 | 作用 |
|---|---|
| `GET` / `PUT /api/monitor/config` | 规则开关与阈值、推送渠道（webhook 只返回脱敏后的地址）、定时设置 |
| `POST /api/monitor/state` | 浏览器同步持仓、自选、候选池和组合参数（前端自动调用） |
| `POST /api/monitor/run` `{rules?, push, force}` | 立即检查 |
| `POST /api/monitor/test` `{channel?}` | 测试推送 |
| `GET` / `DELETE /api/monitor/events` | 事件记录 |

- 定时任务运行在 API 进程内：工作日盘中每 N 分钟检查涨跌停、区制和数据源，收盘后（默认 15:10）运行全部规则。
- 配置、同步的持仓和事件保存在 `state/monitor.json`：已被 git 忽略，文件权限 600。
- Webhook 也可以写在 `.env`（见 `.env.example`），`.env` 中的配置优先于页面填写的值。

## 回测与因子研究

| 接口 | 作用 |
|---|---|
| `POST /api/backtest/run` | 提交回测（策略列表、股票池、区间、调仓频率、成交与成本参数），返回 job id |
| `GET /api/backtest/jobs/{id}` | 进度与结果 |
| `GET /api/backtest/jobs/{id}/report.html` / `returns.csv` | 导出报告 / 日收益 |
| `GET /api/factors/summary?universe=&neutral=&h=` | 全部因子的 IC / ICIR / t / 多空 / 换手 |
| `GET /api/factors/detail/{name}` | 单因子的 IC 序列、分层收益、衰减、逐年 IC |
| `POST /api/factors/corr` | 因子截面秩相关矩阵 |

数据准备：

```bash
python scripts/build_alpha_factors.py   # 因子库 + 月末基础表
python scripts/export_lab.py            # 论文实验室数据
```

`POST /api/portfolio/optimize` 的 `objective` 可取 `minvar`、`rp`、`hrp`、`mv`；`mv` 另有 `risk_aversion` 参数。
