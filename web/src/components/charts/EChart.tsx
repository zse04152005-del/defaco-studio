import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import { useMemo } from "react";
import { cssVar, useUI } from "../../lib/store";

/** Theme-aware ECharts wrapper: axis, grid, tooltip and text pick up the design tokens. */
export function EChart({
  option,
  height = 260,
  onClick,
}: {
  option: EChartsOption;
  height?: number | string;
  onClick?: (p: { name?: string; data?: unknown; seriesName?: string }) => void;
}) {
  const theme = useUI((s) => s.theme);
  const upDown = useUI((s) => s.upDown);
  const merged = useMemo<EChartsOption>(() => {
    const text = cssVar("--text");
    const muted = cssVar("--muted");
    const grid = cssVar("--grid");
    const panel = cssVar("--panel-2");
    const border = cssVar("--border-2");
    const axis = {
      axisLine: { lineStyle: { color: grid } },
      axisTick: { show: false },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: grid } },
    };
    const fix = (a: unknown) =>
      Array.isArray(a) ? a.map((x) => ({ ...axis, ...x })) : a ? { ...axis, ...(a as object) } : undefined;
    return {
      ...option,
      backgroundColor: "transparent",
      textStyle: { color: text, fontFamily: "Inter, PingFang SC, system-ui" },
      animationDuration: 500,
      tooltip: {
        backgroundColor: panel,
        borderColor: border,
        textStyle: { color: text, fontSize: 12 },
        extraCssText: "border-radius:10px;box-shadow:0 8px 24px -12px rgba(0,0,0,.6);",
        ...(option.tooltip as object),
      },
      legend: option.legend
        ? { textStyle: { color: muted, fontSize: 11 }, inactiveColor: grid, pageTextStyle: { color: muted }, ...(option.legend as object) }
        : undefined,
      xAxis: fix(option.xAxis),
      yAxis: fix(option.yAxis),
    } as EChartsOption;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [option, theme, upDown]);
  return (
    <ReactECharts
      option={merged}
      notMerge
      style={{ height, width: "100%" }}
      onEvents={onClick ? { click: onClick } : undefined}
      opts={{ renderer: "canvas" }}
    />
  );
}
