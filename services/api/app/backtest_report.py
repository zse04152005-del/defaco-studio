"""Self-contained HTML backtest report (charts embedded as PNG; prints cleanly to PDF)."""
from __future__ import annotations

import base64
import datetime as dt
import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"font.sans-serif": ["Hiragino Sans GB", "PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
PALETTE = ["#3b6fe0", "#e0852a", "#2e9d6a", "#8b5cc7", "#d14b4b", "#7a8699", "#1aa3a3"]


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>'


def _pct(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:.{d}f}%"


def _num(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


ROWS = [
    ("年化收益", "ann_return", _pct), ("年化波动", "ann_vol", _pct), ("Sharpe（年化，扣无风险利率）", "sharpe", _num),
    ("Sharpe 95% 置信区间", "sharpe_ci95", lambda v: f"[{v[0]:.2f}, {v[1]:.2f}]"),
    ("PSR（Sharpe>0 的概率）", "psr_0", lambda v: _pct(v, 1)), ("DSR（多重比较校正后）", "dsr", lambda v: _pct(v, 1)),
    ("Sortino", "sortino", _num), ("Calmar", "calmar", _num), ("最大回撤", "max_drawdown", _pct),
    ("回撤区间", "dd_peak", None), ("最长水下（交易日）", "longest_underwater_days", lambda v: f"{v}"),
    ("日 VaR 95%", "var95_daily", _pct), ("日 CVaR 95%", "cvar95_daily", _pct),
    ("月度胜率", "monthly_win_rate", lambda v: _pct(v, 0)), ("偏度 / 峰度", "skew", None),
    ("超额年化（vs 沪深300）", "excess_ann_return", _pct), ("跟踪误差", "tracking_error", _pct),
    ("信息比率 IR", "information_ratio", _num), ("Beta", "beta", _num), ("年化 Alpha", "alpha_ann", _pct),
    ("上涨 / 下跌捕获", "up_capture", None),
    ("年换手（单边）", "annual_turnover", lambda v: f"{v:.2f} 倍"), ("交易成本拖累（年化）", "cost_drag_ann", _pct),
    ("平均持股数", "avg_holdings", lambda v: f"{v:.0f}"), ("未成交比例（每次调仓平均）", "unfilled_avg", _pct),
]


def _cell(m: dict, key: str, f) -> str:
    if key == "dd_peak":
        return f"{m.get('dd_peak')} → {m.get('dd_trough')}<br><small>{'恢复于 ' + m['dd_recovered'] if m.get('dd_recovered') else '尚未恢复'}</small>"
    if key == "skew":
        return f"{_num(m.get('skew'))} / {_num(m.get('kurtosis'))}"
    if key == "up_capture":
        return f"{_num(m.get('up_capture'))} / {_num(m.get('down_capture'))}"
    v = m.get(key)
    if v is None:
        return "—"
    return html.escape(f(v)) if f else html.escape(str(v))


def build_report(res: dict) -> str:
    S = res["strategies"]
    names = list(S)
    rets = {k: v["_returns"] for k, v in S.items()}
    bench = None
    if "benchmark" in res:
        bench = pd.Series({pd.Timestamp(d): v for d, v in res["benchmark"]["nav"]})

    # --- charts
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, k in enumerate(names):
        nav = (1 + rets[k]).cumprod()
        ax.plot(nav.index, nav, lw=1.6 if i == 0 else 1.1, color=PALETTE[i % 7], label=k)
    if bench is not None:
        ax.plot(bench.index, bench, lw=1, ls="--", color="#555", label="沪深300")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.set_title("净值（对数坐标）")
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
    nav_png = _png(fig)

    fig, ax = plt.subplots(figsize=(10, 2.4))
    for i, k in enumerate(names):
        nav = (1 + rets[k]).cumprod()
        ax.plot(nav.index, nav / nav.cummax() - 1, lw=1, color=PALETTE[i % 7], label=k)
    ax.set_title("回撤")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    dd_png = _png(fig)

    fig, ax = plt.subplots(figsize=(10, 2.4))
    for i, k in enumerate(names):
        r = rets[k]
        roll = r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252)
        ax.plot(roll.index, roll, lw=1, color=PALETTE[i % 7], label=k)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_title("滚动 12 个月 Sharpe（未扣无风险利率）")
    roll_png = _png(fig)

    years = sorted({d.year for d in rets[names[0]].index})
    fig, ax = plt.subplots(figsize=(10, 2.8))
    width = 0.8 / (len(names) + (bench is not None))
    for i, k in enumerate(names):
        y = (1 + rets[k]).groupby(rets[k].index.year).prod() - 1
        ax.bar(np.arange(len(years)) + i * width, y.reindex(years).values, width, color=PALETTE[i % 7], label=k)
    if bench is not None:
        by = bench.groupby(bench.index.year).last() / bench.groupby(bench.index.year).last().shift(1).fillna(1.0) - 1
        ax.bar(np.arange(len(years)) + len(names) * width, by.reindex(years).values, width, color="#bbb", label="沪深300")
    ax.set_xticks(np.arange(len(years)) + width * len(names) / 2, years)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("逐年收益", pad=22)
    ax.legend(fontsize=8, ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    year_png = _png(fig)

    heat = []
    for k in names:
        m = pd.Series(S[k]["monthly"])
        tab = pd.DataFrame({"y": [s[:4] for s in m.index], "m": [int(s[5:7]) for s in m.index], "v": m.values}).pivot(index="y", columns="m", values="v")
        fig, ax = plt.subplots(figsize=(10, 0.28 * len(tab) + 0.8))
        lim = max(0.05, float(np.nanmax(np.abs(tab.values))))
        ax.imshow(tab.values, cmap="RdYlGn_r", vmin=-lim, vmax=lim, aspect="auto")
        for (i, j), v in np.ndenumerate(tab.values):
            if np.isfinite(v):
                ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(tab.shape[1]), [f"{c}月" for c in tab.columns])
        ax.set_yticks(range(len(tab)), tab.index)
        ax.grid(False)
        ax.set_title(f"月度收益（%）· {k}")
        heat.append(_png(fig))

    # --- tables
    head = "".join(f"<th>{html.escape(k)}</th>" for k in names)
    rows = "".join(f"<tr><td>{lab}</td>" + "".join(f"<td>{_cell(S[k]['metrics'], key, f)}</td>" for k in names) + "</tr>" for lab, key, f in ROWS)
    cost_rows = ""
    for k in names:
        cb = S[k]["metrics"].get("cost_breakdown_ann", {})
        blk = S[k]["metrics"].get("blocked_orders", {})
        cost_rows += (f"<tr><td>{html.escape(k)}</td><td>{_pct(cb.get('commission'), 3)}</td><td>{_pct(cb.get('stamp'), 3)}</td>"
                      f"<td>{_pct(cb.get('transfer'), 3)}</td><td>{_pct(cb.get('impact'), 3)}</td>"
                      f"<td>{'；'.join(f'{a} {b}' for a, b in blk.items()) or '—'}</td></tr>")
    ex = res["execution"]
    timing = "信号日收盘后生成订单，次一交易日开盘成交（T+1）" if ex["timing"] == "next_open" else "信号日收盘价成交（与基准论文一致的理想化设定）"
    cap = "不限" if ex["participation_cap"] is None else f"{ex['participation_cap']:.0%}"
    best = max(names, key=lambda k: S[k]["metrics"].get("dsr") or 0)
    dsr_note = (f"本次共比较 {S[names[0]]['metrics'].get('dsr_trials')} 个策略；在“这些策略都没有真实能力”的假设下，"
                f"最好的一个也会有约 {S[names[0]]['metrics'].get('sr0_ann', 0):.2f} 的年化 Sharpe。DSR 衡量的是超过这一门槛的概率。")
    sig = [k for k in names if (S[k]["metrics"].get("sharpe_ci95") or [0])[0] > 0]
    sig_note = ("、".join(sig) + " 的 Sharpe 95% 置信区间不含 0。") if sig else "所有策略的 Sharpe 95% 置信区间都包含 0：在这段样本里，收益优势在统计上不显著，结论应以风险指标（波动、回撤）为主。"

    css = """body{font-family:-apple-system,'PingFang SC','Hiragino Sans GB',sans-serif;color:#1d2330;max-width:1060px;margin:28px auto;padding:0 20px;font-size:13px;line-height:1.6}
h1{font-size:22px;margin:0}h2{font-size:16px;margin:28px 0 8px;border-left:4px solid #3b6fe0;padding-left:8px}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #e3e6ec;padding:5px 8px;text-align:right}th{background:#f4f6fa}
td:first-child,th:first-child{text-align:left}img{width:100%;margin:6px 0}.box{background:#f6f8fc;border:1px solid #e3e6ec;border-radius:8px;padding:10px 14px}
.muted{color:#6b7385;font-size:12px}small{color:#6b7385}@media print{h2{break-after:avoid}img{break-inside:avoid}}"""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeFaCo 回测报告</title><style>{css}</style></head><body>
<h1>DeFaCo Studio 回测报告</h1>
<p class="muted">生成于 {dt.datetime.now():%Y-%m-%d %H:%M} · 数据集 {html.escape(res['dataset'])} · 区间 {res['start']} → {res['end']} · {res['freq']}调仓 {res['rebalances']} 次 · 股票池：{html.escape(res['universe'])}</p>
<div class="box"><b>结论速览</b>：按 DSR 最可信的是 <b>{html.escape(best)}</b>。{html.escape(sig_note)}</div>

<h2>1. 净值与回撤</h2>{nav_png}{dd_png}
<h2>2. 绩效与统计检验</h2><table><tr><th>指标</th>{head}</tr>{rows}</table>
<p class="muted">PSR = Sharpe 大于 0 的概率（考虑偏度与峰度，Bailey &amp; López de Prado 2012）。{html.escape(dsr_note)}</p>
<h2>3. 收益的时间分布</h2>{year_png}{roll_png}{''.join(heat)}
<h2>4. 交易成本与成交约束</h2>
<table><tr><th>策略</th><th>佣金</th><th>印花税</th><th>过户费</th><th>冲击成本</th><th>被阻挡的订单（次数）</th></tr>{cost_rows}</table>
<h2>5. 回测设定与方法</h2>
<div class="box"><ul>
<li><b>无前视</b>：策略只使用信号日及以前的数据；风险模型用 250 日窗口，股票池用当时的沪深300 历史成分股（非幸存者样本）。</li>
<li><b>成交</b>：{timing}。停牌不能交易；开盘（或收盘）即涨停不能买、跌停不能卖，订单失败的部分留作现金（参照 Qlib Exchange）；退市股按最后价格清仓。</li>
<li><b>流动性</b>：单笔成交不超过当日成交额的 {cap}，资金规模 {ex['aum'] / 1e8:g} 亿元。</li>
<li><b>成本</b>：佣金 {ex['commission_bps']}bp（每笔最低 {ex['min_commission']:g} 元）；卖出印花税 0.1%（2023-08-28 起 0.05%）；过户费 0.001%；冲击成本 = {ex['impact_y']} × 日波动 × √(成交额 / 当日成交额)（平方根冲击模型）。</li>
<li><b>现金</b>按 SHIBOR 3M 计息；Sharpe、Sortino 均扣除无风险利率；基准为沪深300 价格指数。</li>
<li><b>局限</b>：成交价使用开盘 / 收盘价，没有盘口与排队；分红按日收益率中的除权处理；10 年样本的统计功效有限。</li>
</ul></div>
<p class="muted">仅供研究与教学，不构成投资建议。</p>
</body></html>"""
