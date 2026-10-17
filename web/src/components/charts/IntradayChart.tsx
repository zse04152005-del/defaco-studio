// 分时图: price vs previous close, VWAP line and per-minute volume (Lightweight Charts).
import { useEffect, useRef } from "react";
import { createChart, LineStyle, type Time, type UTCTimestamp } from "lightweight-charts";
import type { MinuteBar } from "../../lib/realtime";
import { cssVar, useUI } from "../../lib/store";

const toTs = (t: string) => (Date.parse(t.replace(" ", "T") + ":00Z") / 1000) as UTCTimestamp;

export function IntradayChart({ bars, prevClose, height = 460 }: { bars: MinuteBar[]; prevClose: number | null; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const theme = useUI((s) => s.theme);
  const upDown = useUI((s) => s.upDown);

  useEffect(() => {
    if (!ref.current || !bars.length) return;
    const up = cssVar("--up"), down = cssVar("--down"), muted = cssVar("--muted"), grid = cssVar("--grid"), border = cssVar("--border");
    const pts = bars.filter((b) => b.t);
    const last = pts[pts.length - 1]?.p ?? prevClose ?? 0;
    const color = prevClose == null || last >= prevClose ? up : down;
    const chart = createChart(ref.current, {
      autoSize: true,
      height,
      layout: { background: { color: "transparent" }, textColor: muted, fontSize: 11, fontFamily: "Inter, PingFang SC, system-ui" },
      grid: { vertLines: { color: grid }, horzLines: { color: grid } },
      rightPriceScale: { borderColor: border, scaleMargins: { top: 0.08, bottom: 0.28 } },
      timeScale: { borderColor: border, timeVisible: true, secondsVisible: false },
      localization: { locale: "zh-CN" },
    });
    const price = chart.addAreaSeries({
      lineColor: color, lineWidth: 2, topColor: color + "44", bottomColor: color + "00", priceLineVisible: false,
    });
    price.setData(pts.map((b) => ({ time: toTs(b.t!) as Time, value: b.p })));
    if (prevClose != null) {
      price.createPriceLine({ price: prevClose, color: cssVar("--faint"), lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: "昨收" });
    }
    const avg = chart.addLineSeries({ color: "#f5a524", lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    avg.setData(pts.map((b) => ({ time: toTs(b.t!) as Time, value: b.avg })));
    const vol = chart.addHistogramSeries({ priceScaleId: "vol", priceFormat: { type: "volume" }, lastValueVisible: false, priceLineVisible: false });
    chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.78, bottom: 0 } });
    vol.setData(pts.map((b, i) => ({ time: toTs(b.t!) as Time, value: b.v, color: (i && b.p < pts[i - 1].p ? down : up) + "99" })));
    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [bars, prevClose, height, theme, upDown]);

  return (
    <div>
      <div ref={ref} style={{ height }} />
      <div className="mt-1 flex items-center justify-between text-[10px] text-faint">
        <span className="flex items-center gap-3">
          <span className="flex items-center gap-1"><i className="inline-block h-0.5 w-3 rounded bg-[#f5a524]" />均价</span>
          <span className="flex items-center gap-1"><i className="inline-block h-px w-3 border-t border-dashed border-faint" />昨收</span>
        </span>
        <span>Charts by TradingView Lightweight Charts™</span>
      </div>
    </div>
  );
}

export function OrderBook({ bids, asks, prevClose }: { bids: [number | null, number][]; asks: [number | null, number][]; prevClose: number | null }) {
  const max = Math.max(1, ...bids.map((b) => b[1] || 0), ...asks.map((a) => a[1] || 0));
  const row = (label: string, p: number | null, v: number, side: "ask" | "bid") => (
    <div key={label} className="relative grid grid-cols-[34px_1fr_1fr] items-center px-1 py-[3px] text-xs">
      <div
        className="absolute inset-y-0.5 right-0 rounded-sm"
        style={{ width: `${(v / max) * 60}%`, background: side === "ask" ? "color-mix(in srgb, var(--down) 14%, transparent)" : "color-mix(in srgb, var(--up) 14%, transparent)" }}
      />
      <span className="relative text-faint">{label}</span>
      <span className={`num relative ${p == null || prevClose == null ? "" : p > prevClose ? "text-up" : p < prevClose ? "text-down" : ""}`}>{p ? p.toFixed(2) : "—"}</span>
      <span className="num relative text-right text-muted">{v ? Math.round(v / 100) : "—"}</span>
    </div>
  );
  return (
    <div>
      {[...asks].reverse().map((a, i) => row(`卖${5 - i}`, a[0], a[1], "ask"))}
      <div className="my-1 border-t border-line" />
      {bids.map((b, i) => row(`买${i + 1}`, b[0], b[1], "bid"))}
      <div className="mt-1 text-right text-[10px] text-faint">单位：手</div>
    </div>
  );
}
