import { CURVE, MIX } from "@/app/lib/model";

type MixItem = { name: string; pct: number; tone: string; src: string };

/* ---------- Data-mix donut ---------- */
export function DonutMix({
  mix = MIX,
  centerValue = "2.04B",
  centerLabel = "TOKENS",
}: {
  mix?: readonly MixItem[];
  centerValue?: string;
  centerLabel?: string;
}) {
  const r = 66;
  const cx = 90;
  const cy = 90;
  const C = 2 * Math.PI * r;
  let offset = 0;
  const arcs = mix.map((m) => {
    const len = (m.pct / 100) * C;
    const seg = {
      tone: m.tone,
      dash: `${len} ${C - len}`,
      rot: (offset / C) * 360 - 90,
    };
    offset += len;
    return seg;
  });

  return (
    <div style={{ display: "flex", gap: "2rem", alignItems: "center", flexWrap: "wrap" }}>
      <svg width="180" height="180" viewBox="0 0 180 180" role="img" aria-label="Training data composition">
        {arcs.map((a, i) => (
          <circle
            key={i}
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke={a.tone}
            strokeWidth="16"
            strokeDasharray={a.dash}
            transform={`rotate(${a.rot} ${cx} ${cy})`}
            strokeLinecap="butt"
            style={{ transition: "stroke-dasharray 1s ease" }}
          />
        ))}
        <text x={cx} y={cy - 4} textAnchor="middle" className="stat-num" fontSize="24" fill="var(--ink)">
          {centerValue}
        </text>
        <text x={cx} y={cy + 16} textAnchor="middle" fontSize="9" letterSpacing="0.14em" fill="var(--faint)" fontFamily="var(--font-mono)">
          {centerLabel}
        </text>
      </svg>
      <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "grid", gap: "0.9rem" }}>
        {mix.map((m) => (
          <li key={m.name} style={{ display: "flex", alignItems: "baseline", gap: "0.7rem" }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: m.tone, flexShrink: 0, transform: "translateY(1px)" }} />
            <span>
              <span className="stat-num" style={{ fontSize: "1.25rem" }}>{m.pct}%</span>
              <span style={{ marginLeft: 8, color: "var(--ink)" }}>{m.name}</span>
              <span className="mono" style={{ display: "block", fontSize: "0.72rem", color: "var(--faint)", marginTop: 1 }}>
                {m.src}
              </span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ---------- Training perplexity curve ---------- */
export function TrainingCurve() {
  const W = 640;
  const H = 300;
  const pad = { t: 24, r: 20, b: 40, l: 48 };
  const xs = CURVE.map((d) => d.step);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = 7;
  const ymax = 17;
  const px = (s: number) => pad.l + ((s - xmin) / (xmax - xmin)) * (W - pad.l - pad.r);
  const py = (p: number) => pad.t + (1 - (p - ymin) / (ymax - ymin)) * (H - pad.t - pad.b);

  const line = CURVE.map((d, i) => `${i === 0 ? "M" : "L"} ${px(d.step).toFixed(1)} ${py(d.ppl).toFixed(1)}`).join(" ");
  const area = `${line} L ${px(xmax).toFixed(1)} ${py(ymin).toFixed(1)} L ${px(xmin).toFixed(1)} ${py(ymin).toFixed(1)} Z`;
  const gridY = [7, 9, 11, 13, 15, 17];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Held-out perplexity over training steps" style={{ maxWidth: W }}>
      <defs>
        <linearGradient id="fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--green)" stopOpacity="0.16" />
          <stop offset="100%" stopColor="var(--green)" stopOpacity="0" />
        </linearGradient>
      </defs>

      {gridY.map((g) => (
        <g key={g}>
          <line x1={pad.l} y1={py(g)} x2={W - pad.r} y2={py(g)} stroke="var(--line)" strokeWidth="1" />
          <text x={pad.l - 10} y={py(g) + 3} textAnchor="end" fontSize="10" fill="var(--faint)" fontFamily="var(--font-mono)">
            {g}
          </text>
        </g>
      ))}

      <path d={area} fill="url(#fill)" />
      <path
        d={line}
        fill="none"
        stroke="var(--green)"
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
        style={{ strokeDasharray: 1600, strokeDashoffset: 1600, animation: "drawLine 1.8s ease forwards 0.2s" }}
      />

      {CURVE.map((d, i) => (
        <circle key={i} cx={px(d.step)} cy={py(d.ppl)} r={i === CURVE.length - 1 ? 4.5 : 3} fill="var(--paper-2)" stroke="var(--green)" strokeWidth="2" />
      ))}

      {/* final annotation */}
      <text x={px(xmax) - 6} y={py(CURVE[CURVE.length - 1].ppl) - 14} textAnchor="end" className="stat-num" fontSize="16" fill="var(--green)">
        {CURVE[CURVE.length - 1].ppl}
      </text>

      {/* x labels */}
      {[1000, 20000, xmax].map((s) => (
        <text key={s} x={px(s)} y={H - 14} textAnchor="middle" fontSize="10" fill="var(--faint)" fontFamily="var(--font-mono)">
          {s === xmax ? `${(s / 1000).toFixed(1)}k` : `${s / 1000}k`}
        </text>
      ))}
      <text x={(W + pad.l) / 2} y={H - 1} textAnchor="middle" fontSize="9.5" letterSpacing="0.16em" fill="var(--faint)" fontFamily="var(--font-mono)">
        OPTIMIZER STEP →
      </text>
    </svg>
  );
}

/* ---------- Per-source perplexity, 500M vs 125M ---------- */
export function PerSourcePplBars({
  data,
}: {
  data: readonly { source: string; ppl500: number; ppl125: number | null }[];
}) {
  const max = Math.max(...data.map((d) => Math.max(d.ppl500, d.ppl125 ?? 0))) * 1.08;
  return (
    <div style={{ display: "grid", gap: "1.1rem" }}>
      {data.map((d) => (
        <div key={d.source}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "0.4rem" }}>
            <span style={{ fontSize: "0.88rem", color: "var(--ink)" }}>{d.source}</span>
            <span className="mono tnum" style={{ fontSize: "0.76rem", color: "var(--faint)" }}>
              {d.ppl125 !== null ? `${d.ppl125} → ` : ""}
              <span style={{ color: "var(--green)" }}>{d.ppl500}</span>
            </span>
          </div>
          <div style={{ display: "grid", gap: "3px" }}>
            {d.ppl125 !== null && (
              <div style={{ height: 8, borderRadius: 2, background: "var(--paper-3)", border: "1px solid var(--line-2)", position: "relative", overflow: "hidden" }}>
                <div style={{ position: "absolute", inset: 0, width: `${(d.ppl125 / max) * 100}%`, background: "var(--faint)", opacity: 0.45 }} />
              </div>
            )}
            <div style={{ height: 8, borderRadius: 2, background: "var(--paper-3)", border: "1px solid var(--line-2)", position: "relative", overflow: "hidden" }}>
              <div style={{ position: "absolute", inset: 0, width: `${(d.ppl500 / max) * 100}%`, background: "var(--green)" }} />
            </div>
          </div>
        </div>
      ))}
      <div style={{ display: "flex", gap: "1.25rem", fontSize: "0.72rem", color: "var(--faint)", marginTop: "0.3rem" }}>
        <span><span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: "var(--faint)", opacity: 0.45, marginRight: 6, transform: "translateY(1px)" }} />125M</span>
        <span><span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: "var(--green)", marginRight: 6, transform: "translateY(1px)" }} />500M</span>
      </div>
    </div>
  );
}
