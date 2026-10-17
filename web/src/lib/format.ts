export const pct = (x: number | null | undefined, d = 2) =>
  x == null || Number.isNaN(x) ? "—" : `${(x * 100).toFixed(d)}%`;

export const signedPct = (x: number | null | undefined, d = 2) =>
  x == null || Number.isNaN(x) ? "—" : `${x > 0 ? "+" : ""}${(x * 100).toFixed(d)}%`;

export const num = (x: number | null | undefined, d = 2) =>
  x == null || Number.isNaN(x) ? "—" : x.toLocaleString("zh-CN", { minimumFractionDigits: d, maximumFractionDigits: d });

export const yi = (x: number | null | undefined) => {
  if (x == null || Number.isNaN(x)) return "—";
  return x >= 10000 ? `${(x / 10000).toFixed(2)} 万亿` : `${x.toFixed(0)} 亿`;
};

/** CSS colour token for a signed number (A-share red-up by default, switchable). */
export const tone = (x: number | null | undefined) =>
  x == null || x === 0 ? "text-muted" : x > 0 ? "text-up" : "text-down";

export const arrow = (x: number | null | undefined) => (x == null || x === 0 ? "" : x > 0 ? "▲" : "▼");
