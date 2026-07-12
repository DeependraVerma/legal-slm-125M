"use client";

import { useEffect, useRef, useState } from "react";
import { CHAT_PRESETS, CHAT_URL } from "@/app/lib/model";

type Role = "user" | "assistant";
type Msg = { role: Role; content: string };
type Status = "idle" | "waking" | "streaming";

export default function Chat() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const busy = status === "waking" || status === "streaming";

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, status]);

  async function send(text: string) {
    const message = text.trim();
    if (!message || busy) return;
    setError("");
    setInput("");
    setMessages((m) => [...m, { role: "user", content: message }, { role: "assistant", content: "" }]);
    setStatus("waking");
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const res = await fetch(`${CHAT_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, max_new_tokens: 200, temperature: 0.7 }),
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
          try {
            const obj = JSON.parse(line.slice(5).trim());
            if (obj.token) {
              if (first) {
                first = false;
                setStatus("streaming");
              }
              setMessages((m) => {
                const copy = m.slice();
                copy[copy.length - 1] = {
                  role: "assistant",
                  content: copy[copy.length - 1].content + obj.token,
                };
                return copy;
              });
            }
          } catch {
            /* ignore */
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setError("Could not reach the model — it may be waking up. Try again in a moment.");
        setMessages((m) => m.filter((_, i) => !(i === m.length - 1 && m[i].role === "assistant" && m[i].content === "")));
      }
    } finally {
      setStatus("idle");
    }
  }

  function reset() {
    abortRef.current?.abort();
    setMessages([]);
    setError("");
    setStatus("idle");
  }

  return (
    <div className="paper-card" style={{ padding: "clamp(1.1rem, 2.5vw, 1.75rem)", display: "flex", flexDirection: "column" }}>
      {/* header row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
        <span className="eyebrow">Ask the assistant</span>
        {messages.length > 0 && (
          <button onClick={reset} className="mono" style={{ fontSize: "0.72rem", color: "var(--faint)", background: "none", border: "none", cursor: "pointer" }}>
            new chat ↺
          </button>
        )}
      </div>

      {/* messages */}
      <div
        ref={scrollRef}
        style={{
          minHeight: "16rem",
          maxHeight: "26rem",
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: "1rem",
          padding: "0.25rem",
        }}
      >
        {messages.length === 0 && (
          <div style={{ margin: "auto", textAlign: "center", maxWidth: "32ch" }}>
            <p style={{ color: "var(--faint)", fontSize: "0.92rem", lineHeight: 1.6 }}>
              Ask a legal or financial question. The fine-tuned model answers — pick a starter below or type your own.
            </p>
          </div>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <Bubble key={i} side="right">{m.content}</Bubble>
          ) : (
            <Bubble key={i} side="left" mono>
              {m.content ? m.content : (status === "waking" ? <span className="shimmer" style={{ display: "inline-block", width: "9rem", height: "0.9em", borderRadius: 3 }} /> : "")}
              {i === messages.length - 1 && busy && m.content && <span className="caret" />}
            </Bubble>
          )
        )}
      </div>

      {error && (
        <p style={{ margin: "0.75rem 0 0", color: "var(--brass)", fontSize: "0.82rem" }}>{error}</p>
      )}

      {/* presets */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.45rem", margin: "1.1rem 0 0.85rem" }}>
        {CHAT_PRESETS.map((p) => (
          <button key={p} className="chip" onClick={() => send(p)} disabled={busy}>
            {p.length > 40 ? p.slice(0, 38) + "…" : p}
          </button>
        ))}
      </div>

      {/* input */}
      <form
        onSubmit={(e) => { e.preventDefault(); send(input); }}
        style={{ display: "flex", gap: "0.6rem", alignItems: "flex-end" }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); }
          }}
          rows={1}
          placeholder="Ask a legal or financial question…"
          spellCheck={false}
          style={{
            flex: 1,
            resize: "none",
            background: "var(--paper-3)",
            border: "1px solid var(--line-2)",
            borderRadius: 4,
            padding: "0.75rem 0.9rem",
            fontFamily: "var(--font-sans)",
            fontSize: "0.95rem",
            color: "var(--ink)",
            lineHeight: 1.5,
            outline: "none",
            maxHeight: "8rem",
          }}
        />
        <button className="btn-primary" type="submit" disabled={busy || !input.trim()}>
          {busy ? "…" : "Send →"}
        </button>
      </form>

      <p style={{ marginTop: "0.9rem", fontSize: "0.8rem", color: "var(--faint)", lineHeight: 1.6 }}>
        A 125M fine-tuned model — it answers one question at a time and will confidently
        invent case names and figures. Not legal or financial advice.
      </p>
    </div>
  );
}

function Bubble({ side, mono, children }: { side: "left" | "right"; mono?: boolean; children: React.ReactNode }) {
  const isRight = side === "right";
  return (
    <div style={{ display: "flex", justifyContent: isRight ? "flex-end" : "flex-start" }}>
      <div
        style={{
          maxWidth: "82%",
          padding: "0.7rem 0.95rem",
          borderRadius: 8,
          fontSize: mono ? "0.9rem" : "0.95rem",
          lineHeight: 1.6,
          whiteSpace: "pre-wrap",
          fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)",
          background: isRight ? "var(--paper-3)" : "var(--paper-2)",
          border: `1px solid ${isRight ? "var(--line-2)" : "var(--line)"}`,
          borderLeft: isRight ? undefined : "2px solid var(--green)",
          color: "var(--ink)",
        }}
      >
        <span className="mono" style={{ display: "block", fontSize: "0.62rem", letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--faint)", marginBottom: "0.3rem" }}>
          {isRight ? "you" : "assistant"}
        </span>
        {children}
      </div>
    </div>
  );
}
