import clsx from "clsx";
import type { ReactNode } from "react";
import { Info } from "lucide-react";
import { arrow, tone } from "../lib/format";

/* ------------------------------------------------------------------ layout */
export function Panel({
  title,
  subtitle,
  actions,
  children,
  className,
  bodyClass,
  accent,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
  accent?: string; // module colour bar, e.g. "var(--m-cil)"
}) {
  return (
    <section className={clsx("panel flex min-w-0 flex-col fade-up", className)}>
      {(title || actions) && (
        <header className="flex items-center gap-3 border-b border-line px-4 py-2.5">
          {accent && <span className="h-4 w-1 rounded-full" style={{ background: accent }} />}
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-[13px] font-semibold tracking-wide text-ink">{title}</h3>
            {subtitle && <p className="truncate text-[11px] text-faint">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={clsx("min-h-0 flex-1 p-4", bodyClass)}>{children}</div>
    </section>
  );
}

export function PageHeader({ title, desc, actions }: { title: string; desc?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-4 flex flex-wrap items-end gap-3">
      <div className="min-w-0 flex-1">
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {desc && <div className="mt-0.5 text-[13px] text-muted">{desc}</div>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ data display */
export function Stat({
  label,
  value,
  hint,
  delta,
  className,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  delta?: number | null;
  className?: string;
}) {
  return (
    <div className={clsx("rounded-xl border border-line bg-panel-2/60 px-3.5 py-3", className)}>
      <div className="text-[11px] text-faint">{label}</div>
      <div className="num mt-1 text-lg font-semibold leading-tight">{value}</div>
      {(hint || delta != null) && (
        <div className="mt-0.5 flex items-center gap-2 text-[11px] text-muted">
          {delta != null && <Delta value={delta} />}
          {hint}
        </div>
      )}
    </div>
  );
}

/** Signed change: colour + ▲▼ symbol (colour-blind safe). */
export function Delta({ value, digits = 2, className }: { value: number | null | undefined; digits?: number; className?: string }) {
  if (value == null || Number.isNaN(value)) return <span className="text-muted">—</span>;
  return (
    <span className={clsx("num", tone(value), className)}>
      {arrow(value)} {value > 0 ? "+" : ""}
      {(value * 100).toFixed(digits)}%
    </span>
  );
}

export function Badge({ children, color = "var(--accent)" }: { children: ReactNode; color?: string }) {
  return (
    <span
      className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium"
      style={{ color, background: `color-mix(in srgb, ${color} 14%, transparent)` }}
    >
      {children}
    </span>
  );
}

export function Empty({ title, desc, icon }: { title: string; desc?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex h-full min-h-40 flex-col items-center justify-center gap-2 text-center">
      {icon}
      <div className="text-sm font-medium text-ink">{title}</div>
      {desc && <div className="max-w-sm text-xs text-muted">{desc}</div>}
    </div>
  );
}

export function DataBanner({ note }: { note?: string }) {
  if (!note) return null;
  return (
    <div className="mb-4 flex items-center gap-2 rounded-xl border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn">
      <Info size={14} /> {note}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("skeleton", className)} />;
}

/* ------------------------------------------------------------------ Uiverse-based controls */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  name,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: ReactNode }[];
  name: string;
}) {
  return (
    <div className="uv-seg" role="radiogroup">
      {options.map((o) => (
        <label key={o.value}>
          <input type="radio" name={name} checked={value === o.value} onChange={() => onChange(o.value)} />
          <span>{o.label}</span>
        </label>
      ))}
    </div>
  );
}

export function GlowButton({ children, onClick, disabled }: { children: ReactNode; onClick?: () => void; disabled?: boolean }) {
  return (
    <button className="uv-glow" onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}

export function AiButton({ children = "AI 解读", onClick }: { children?: ReactNode; onClick?: () => void }) {
  return (
    <button className="uv-ai" onClick={onClick}>
      <svg height="16" width="16" viewBox="0 0 24 24" className="uv-ai-sparkle" aria-hidden>
        <path d="M10,21.236,6.755,14.745.264,11.5,6.755,8.255,10,1.764l3.245,6.491L19.736,11.5l-6.491,3.245ZM18,21l1.5,3L21,21l3-1.5L21,18l-1.5-3L18,18l-3,1.5ZM19.333,4.667,20.5,7l1.167-2.333L24,3.5,21.667,2.333,20.5,0,19.333,2.333,17,3.5Z" />
      </svg>
      <span className="uv-ai-text">{children}</span>
    </button>
  );
}

export function RimButton({ children, onClick, disabled, title }: { children: ReactNode; onClick?: () => void; disabled?: boolean; title?: string }) {
  return (
    <div className={clsx("uv-rim", disabled && "pointer-events-none opacity-40")} title={title}>
      <button onClick={onClick} disabled={disabled}>{children}</button>
    </div>
  );
}

export function UploadButton({ children, onClick }: { children: ReactNode; onClick?: () => void }) {
  return (
    <button className="uv-upload" onClick={onClick}>
      <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden>
        <path fill="#000" d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l4.65-4.65c.2-.2.51-.2.71 0L17 13h-3z" />
      </svg>
      {children}
    </button>
  );
}

export function ThemeSwitch({ light, onChange }: { light: boolean; onChange: (light: boolean) => void }) {
  return (
    <input
      type="checkbox"
      className="uv-theme"
      aria-label="切换浅色/深色主题"
      title={light ? "切换到深色" : "切换到浅色"}
      checked={light}
      onChange={(e) => onChange(e.target.checked)}
    />
  );
}

export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label?: ReactNode }) {
  return (
    <label className="uv-switch inline-flex cursor-pointer items-center gap-2 text-xs text-muted">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <div className="uv-switch-bg">
        <span className="uv-switch-knob" />
      </div>
      {label}
    </label>
  );
}

export function InfinityLoader({ size = 84, label }: { size?: number; label?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 text-xs text-muted">
      <svg className="uv-infinity" viewBox="0 0 187.3 93.7" width={size} height={size / 2} aria-hidden>
        <path
          id="uv-outline"
          d="M93.9,46.4c9.3,9.5,13.8,17.9,23.5,17.9s17.5-7.8,17.5-17.5s-7.8-17.6-17.5-17.5c-9.7,0.1-13.3,7.2-22.1,17.1c-8.9,8.8-15.7,17.9-25.4,17.9s-17.5-7.8-17.5-17.5s7.8-17.5,17.5-17.5S86.2,38.6,93.9,46.4z"
          fill="none"
          stroke="var(--accent)"
          strokeWidth="5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M93.9,46.4c9.3,9.5,13.8,17.9,23.5,17.9s17.5-7.8,17.5-17.5s-7.8-17.6-17.5-17.5c-9.7,0.1-13.3,7.2-22.1,17.1c-8.9,8.8-15.7,17.9-25.4,17.9s-17.5-7.8-17.5-17.5s7.8-17.5,17.5-17.5S86.2,38.6,93.9,46.4z"
          fill="none"
          stroke="var(--accent)"
          strokeWidth="5"
          opacity="0.12"
        />
      </svg>
      {label}
    </div>
  );
}

export function Dots() {
  return (
    <span className="uv-dots" aria-label="加载中">
      <i />
      <i />
      <i />
      <i />
      <i />
    </span>
  );
}

export function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden>
      <path d="M21.53 20.47l-3.66-3.66C19.195 15.24 20 13.214 20 11c0-4.97-4.03-9-9-9s-9 4.03-9 9 4.03 9 9 9c2.215 0 4.24-.804 5.808-2.13l3.66 3.66c.147.146.34.22.53.22s.385-.073.53-.22c.295-.293.295-.767.002-1.06zM3.5 11c0-4.135 3.365-7.5 7.5-7.5s7.5 3.365 7.5 7.5-3.365 7.5-7.5 7.5-7.5-3.365-7.5-7.5z" />
    </svg>
  );
}
