// Professional K-line chart on TradingView Lightweight Charts (Apache-2.0, attribution kept).
import { useEffect, useMemo, useRef, useState } from "react";
import {
  createChart,
  CrosshairMode,
  TickMarkType,
  type IChartApi,
  type ISeriesApi,
  type LogicalRange,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";
import { cssVar, useUI } from "../../lib/store";
import { detectPatterns, kdj, macd, resample, rsi, sma, type Candle } from "../../lib/indicators";
import { Segmented, Switch } from "../ui";

const MA = [
  { n: 5, color: "#f5a524" },
  { n: 10, color: "#4c8dff" },
  { n: 20, color: "#b37feb" },
  { n: 60, color: "#22c3c3" },
];

type Sub = "MACD" | "KDJ" | "RSI";

// Axis labels. The library sizes tick spacing for 8-character labels by default, so a month whose
// first trading day comes < ~18 bars after the previous month tick (e.g. June after the May Day
// holiday) lost its "6月" tick and showed a day tick ("2日") instead. Our labels are at most 5
// characters (tickMarkMaxCharacterLength below), and dates are formatted in UTC.
function toUTC(t: Time): Date {
  if (typeof t === "number") return new Date(t * 1000);
  if (typeof t === "string") return new Date(`${t}T00:00:00Z`);
  return new Date(Date.UTC(t.year, t.month - 1, t.day));
}
const fmtDate = (d: Date) => `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}-${String(d.getUTCDate()).padStart(2, "0")}`;
function tickLabel(t: Time, type: TickMarkType): string {
  const d = toUTC(t);
  switch (type) {
    case TickMarkType.Year:
      return `${d.getUTCFullYear()}年`;
    case TickMarkType.Month:
      return d.getUTCMonth() === 0 ? `${d.getUTCFullYear()}年` : `${d.getUTCMonth() + 1}月`;
    case TickMarkType.DayOfMonth:
      return `${d.getUTCMonth() + 1}/${d.getUTCDate()}`;
    default:
      return `${String(d.getUTCHours()).padStart(2, "0")}:${String(d.getUTCMinutes()).padStart(2, "0")}`;
  }
}

export function KLineChart({ candles, height = 520 }: { candles: Candle[]; height?: number }) {
  const theme = useUI((s) => s.theme);
  const upDown = useUI((s) => s.upDown);
  const [freq, setFreq] = useState<"D" | "W" | "M">("D");
  const [sub, setSub] = useState<Sub>("MACD");
  const [showPatterns, setShowPatterns] = useState(true);
  const [legend, setLegend] = useState<Candle | null>(null);
  const mainRef = useRef<HTMLDivElement>(null);
  const subRef = useRef<HTMLDivElement>(null);

  const data = useMemo(() => resample(candles, freq), [candles, freq]);
  const patterns = useMemo(() => (freq === "D" ? detectPatterns(data) : []), [data, freq]);

  useEffect(() => {
    if (!mainRef.current || !subRef.current || data.length === 0) return;
    const up = cssVar("--up"), down = cssVar("--down"), text = cssVar("--muted"), grid = cssVar("--grid"), border = cssVar("--border");
    const base = {
      layout: { background: { color: "transparent" }, textColor: text, fontSize: 11, fontFamily: "Inter, PingFang SC, system-ui" },
      grid: { vertLines: { color: grid }, horzLines: { color: grid } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: border },
      timeScale: { borderColor: border, rightOffset: 4, barSpacing: 7, tickMarkFormatter: tickLabel, tickMarkMaxCharacterLength: 5 },
      localization: { locale: "zh-CN", timeFormatter: (t: Time) => fmtDate(toUTC(t)) },
    };
    const main: IChartApi = createChart(mainRef.current, { ...base, height: height * 0.68, autoSize: true });
    const subc: IChartApi = createChart(subRef.current, {
      ...base,
      height: height * 0.32,
      autoSize: true,
      timeScale: { ...base.timeScale, visible: true },
    });

    const candle = main.addCandlestickSeries({
      upColor: up, downColor: down, borderUpColor: up, borderDownColor: down, wickUpColor: up, wickDownColor: down,
      priceLineVisible: true,
    });
    candle.setData(data.map((c) => ({ time: c.time as Time, open: c.open, high: c.high, low: c.low, close: c.close })));
    const close = data.map((c) => c.close);
    MA.forEach(({ n, color }) => {
      const s = main.addLineSeries({ color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
      s.setData(sma(close, n).flatMap((v, i) => (v == null ? [] : [{ time: data[i].time as Time, value: v }])));
    });
    const vol = main.addHistogramSeries({ priceScaleId: "vol", priceFormat: { type: "volume" }, lastValueVisible: false, priceLineVisible: false });
    main.priceScale("vol").applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
    vol.setData(data.map((c) => ({ time: c.time as Time, value: c.volume, color: c.close >= c.open ? up + "88" : down + "88" })));

    if (showPatterns && patterns.length) {
      const markers: SeriesMarker<Time>[] = patterns.slice(-12).map((p) => ({
        time: p.time as Time,
        position: p.kind === "bull" ? "belowBar" : "aboveBar",
        color: p.kind === "bull" ? up : down,
        shape: p.kind === "bull" ? "arrowUp" : "arrowDown",
        text: p.name.replace("看涨", "").replace("看跌", ""),
      }));
      candle.setMarkers(markers);
    }

    // sub-indicator pane
    const subSeries: ISeriesApi<"Line" | "Histogram">[] = [];
    if (sub === "MACD") {
      const m = macd(close);
      const h = subc.addHistogramSeries({ priceLineVisible: false, lastValueVisible: false });
      h.setData(m.hist.map((v, i) => ({ time: data[i].time as Time, value: v, color: v >= 0 ? up : down })));
      subSeries.push(h);
      [["dif", "#f5a524"], ["dea", "#4c8dff"]].forEach(([k, color]) => {
        const s = subc.addLineSeries({ color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
        s.setData((m as Record<string, number[]>)[k].map((v, i) => ({ time: data[i].time as Time, value: v })));
        subSeries.push(s);
      });
    } else if (sub === "KDJ") {
      const k = kdj(data);
      [["K", "#f5a524"], ["D", "#4c8dff"], ["J", "#b37feb"]].forEach(([key, color]) => {
        const s = subc.addLineSeries({ color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
        s.setData((k as Record<string, number[]>)[key].map((v, i) => ({ time: data[i].time as Time, value: v })));
        subSeries.push(s);
      });
    } else {
      const r = rsi(close);
      const s = subc.addLineSeries({ color: "#b37feb", lineWidth: 1, priceLineVisible: false });
      s.setData(r.flatMap((v, i) => (v == null ? [] : [{ time: data[i].time as Time, value: v }])));
      [70, 30].forEach((lvl) => s.createPriceLine({ price: lvl, color: cssVar("--faint"), lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "" }));
      subSeries.push(s);
    }

    // show the last ~9 months by default
    const n = data.length;
    main.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - (freq === "D" ? 180 : freq === "W" ? 120 : 60)), to: n + 3 });

    // sync time ranges and crosshair between panes
    let syncing = false;
    const syncRange = (to: IChartApi) => (r: LogicalRange | null) => {
      if (!r || syncing) return;
      syncing = true;
      to.timeScale().setVisibleLogicalRange(r);
      syncing = false;
    };
    main.timeScale().subscribeVisibleLogicalRangeChange(syncRange(subc));
    subc.timeScale().subscribeVisibleLogicalRangeChange(syncRange(main));
    const byTime = new Map(data.map((c) => [c.time, c]));
    main.subscribeCrosshairMove((p) => {
      const c = p.time ? byTime.get(p.time as string) : undefined;
      setLegend(c ?? null);
      if (p.time && subSeries[0]) subc.setCrosshairPosition(0, p.time, subSeries[0]);
      else subc.clearCrosshairPosition();
    });

    return () => {
      main.remove();
      subc.remove();
    };
  }, [data, sub, showPatterns, patterns, theme, upDown, height, freq]);

  const shown = legend ?? data[data.length - 1];
  const prev = shown ? data[data.findIndex((c) => c.time === shown.time) - 1] : undefined;
  const chg = shown && prev ? shown.close / prev.close - 1 : null;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-3">
        <Segmented
          name="kline-freq"
          value={freq}
          onChange={setFreq}
          options={[{ value: "D", label: "日K" }, { value: "W", label: "周K" }, { value: "M", label: "月K" }]}
        />
        <Segmented name="kline-sub" value={sub} onChange={setSub} options={["MACD", "KDJ", "RSI"].map((v) => ({ value: v as Sub, label: v }))} />
        <Switch checked={showPatterns} onChange={setShowPatterns} label="形态标注" />
        <div className="ml-auto flex flex-wrap items-center gap-3 text-[11px] text-muted">
          {MA.map((m) => (
            <span key={m.n} className="flex items-center gap-1">
              <i className="inline-block h-0.5 w-3 rounded" style={{ background: m.color }} />MA{m.n}
            </span>
          ))}
        </div>
      </div>
      {shown && (
        <div className="num flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted">
          <span>{shown.time}</span>
          <span>开 <b className="text-ink">{shown.open.toFixed(2)}</b></span>
          <span>高 <b className="text-ink">{shown.high.toFixed(2)}</b></span>
          <span>低 <b className="text-ink">{shown.low.toFixed(2)}</b></span>
          <span>收 <b className={chg == null ? "text-ink" : chg >= 0 ? "text-up" : "text-down"}>{shown.close.toFixed(2)}</b></span>
          {chg != null && <span className={chg >= 0 ? "text-up" : "text-down"}>{chg >= 0 ? "▲ +" : "▼ "}{(chg * 100).toFixed(2)}%</span>}
          <span>量 {(shown.volume / 1e4).toFixed(0)} 万</span>
          <span>额 {(shown.amount / 1e8).toFixed(2)} 亿</span>
        </div>
      )}
      <div ref={mainRef} style={{ height: height * 0.68 }} />
      <div ref={subRef} style={{ height: height * 0.32 }} className="border-t border-line" />
      <div className="text-right text-[10px] text-faint">Charts by TradingView Lightweight Charts™</div>
    </div>
  );
}
