"use client";

import { useRef, useState } from "react";
import { INFERENCE_URL, PRESETS } from "@/app/lib/model";

type Status = "idle" | "waking" | "streaming" | "done" | "error";

export default function Playground() {
  const [prompt, setPrompt] = useState<string>(PRESETS[0]);
  const [tokens, setTokens] = useState<string[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [maxTokens, setMaxTokens] = useState(96);
  const [temp, setTemp] = useState(0.8);
  const [error, setError] = useState<string>("");
  const abortRef = useRef<AbortController | null>(null);

  const busy = status === "waking" || status === "streaming";

  async function run() {
    if (!prompt.trim() || busy) return;
    setTokens([]);
    setError("");
    setStatus("waking");
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const res = await fetch(`${INFERENCE_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_new_tokens: maxTokens,
          temperature: temp,
        }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) throw new Error(`server ${res.status}`);

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      let first = true;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          try {
            const obj = JSON.parse(payload);
            if (obj.token) {
              if (first) {
                first = false;
                setStatus("streaming");
              }
              setTokens((t) => [...t, obj.token as string]);
            }
            if (obj.done) setStatus("done");
          } catch {
            /* ignore keep-alive / partial */
          }
        }
      }
      setStatus((s) => (s === "streaming" || s === "waking" ? "done" : s));
    } catch (e) {
      if ((e as Error).name === "AbortError") {
        setStatus("done");
      } else {
        setError("Could not reach the model. Please try again in a moment.");
        setStatus("error");
      }
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  return (
    <div className="paper-card" style={{ padding: "clamp(1.25rem, 3vw, 2rem)" }}>
      {/* presets */}
      <div style={{ marginBottom: "1.1rem" }}>
        <div className="eyebrow" style={{ marginBottom: "0.7rem" }}>Starters</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
          {PRESETS.map((p) => (
            <button
              key={p}
              className="chip"
              data-active={prompt === p}
              onClick={() => setPrompt(p)}
              disabled={busy}
            >
              {p.length > 42 ? p.slice(0, 40) + "…" : p}
            </button>
          ))}
        </div>
      </div>

      {/* prompt */}
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={2}
        spellCheck={false}
        placeholder="Write the opening of a legal or financial passage…"
        style={{
          width: "100%",
          resize: "vertical",
          background: "var(--paper-3)",
          border: "1px solid var(--line-2)",
          borderRadius: 4,
          padding: "0.9rem 1rem",
          fontFamily: "var(--font-mono)",
          fontSize: "0.92rem",
          color: "var(--ink)",
          lineHeight: 1.6,
          outline: "none",
        }}
      />

      {/* controls */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.4rem", margin: "1.3rem 0 1.1rem" }}>
        <Control label="Length" value={`${maxTokens} tokens`}>
          <input type="range" min={16} max={200} step={4} value={maxTokens} disabled={busy}
            onChange={(e) => setMaxTokens(+e.target.value)} />
        </Control>
        <Control label="Creativity" value={temp.toFixed(2)}>
          <input type="range" min={0.1} max={1.4} step={0.05} value={temp} disabled={busy}
            onChange={(e) => setTemp(+e.target.value)} />
        </Control>
      </div>

      <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
        {!busy ? (
          <button className="btn-primary" onClick={run} disabled={!prompt.trim()}>
            Complete&nbsp;the&nbsp;passage&nbsp;→
          </button>
        ) : (
          <button className="btn-primary" onClick={stop} style={{ background: "var(--brass)", borderColor: "var(--brass)", boxShadow: "none" }}>
            Stop
          </button>
        )}
        <StatusPill status={status} count={tokens.length} />
      </div>

      {/* output */}
      <div
        aria-live="polite"
        style={{
          marginTop: "1.3rem",
          minHeight: "8.5rem",
          background: "var(--paper-3)",
          border: "1px solid var(--line)",
          borderRadius: 4,
          padding: "1.1rem 1.2rem",
          fontFamily: "var(--font-mono)",
          fontSize: "0.92rem",
          lineHeight: 1.75,
          whiteSpace: "pre-wrap",
          color: "var(--ink)",
        }}
      >
        {status === "idle" && tokens.length === 0 ? (
          <span style={{ color: "var(--faint)" }}>
            The completion will appear here, streamed token by token as the model writes it.
          </span>
        ) : (
          <>
            <span style={{ color: "var(--muted)" }}>{prompt}</span>
            {status === "waking" && tokens.length === 0 && (
              <span className="shimmer" style={{ display: "inline-block", width: "62%", height: "0.95em", borderRadius: 3, marginLeft: 6, transform: "translateY(2px)" }} />
            )}
            {tokens.map((t, i) => (
              <span key={i} className="tok">{t}</span>
            ))}
            {busy && <span className="caret" />}
          </>
        )}
        {error && (
          <span style={{ display: "block", marginTop: "0.8rem", color: "var(--brass)", fontFamily: "var(--font-sans)", fontSize: "0.85rem" }}>
            {error}
          </span>
        )}
      </div>

      <p style={{ marginTop: "1rem", fontSize: "0.82rem", color: "var(--faint)", lineHeight: 1.6 }}>
        This is a <strong style={{ color: "var(--muted)", fontWeight: 500 }}>base model</strong>: it continues text, it does not answer questions.
        It will confidently invent case names, citations and figures. Never rely on its output as legal, financial or factual advice.
      </p>
    </div>
  );
}

function Control({ label, value, children }: { label: string; value: string; children: React.ReactNode }) {
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "0.55rem" }}>
        <span className="eyebrow" style={{ color: "var(--muted)" }}>{label}</span>
        <span className="mono tnum" style={{ fontSize: "0.8rem", color: "var(--green)" }}>{value}</span>
      </div>
      {children}
    </div>
  );
}

function StatusPill({ status, count }: { status: Status; count: number }) {
  const map: Record<Status, string> = {
    idle: "",
    waking: "waking the model…",
    streaming: `writing · ${count} tokens`,
    done: `done · ${count} tokens`,
    error: "unreachable",
  };
  if (!map[status]) return null;
  return (
    <span className="mono" style={{ fontSize: "0.76rem", color: status === "error" ? "var(--brass)" : "var(--faint)" }}>
      {map[status]}
    </span>
  );
}
