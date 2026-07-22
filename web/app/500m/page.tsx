import Chat from "@/app/components/Chat";
import Nav from "@/app/components/Nav";
import { DonutMix, PerSourcePplBars } from "@/app/components/Visuals";
import {
  ARCH,
  CHAT_PRESETS,
  COMPARISON,
  HERO_STATS,
  HF_ONNX_URL,
  HF_SFT_URL,
  HF_URL,
  INFRA_NOTE,
  MIX,
  NUMBERS,
  PERSOURCE_PPL,
  SFT_STATS,
} from "@/app/lib/model500m";
import { GITHUB_URL, HF_PROFILE_URL, JOB_TITLE_LINE, PERSON_NAME } from "@/app/lib/seo";

export const metadata = {
  title: "legal-slm-500M — a 528.5M-parameter legal & financial LLM built from scratch",
  description:
    "A 528.5M-parameter Llama-style language model, trained completely from scratch on a staged " +
    "5-source legal/financial data mix — the scaled-up successor to legal-slm-125M.",
};

const FAQS = [
  {
    q: "What's actually different from the 125M model?",
    a: "More parameters (528.5M vs 125.8M) and more, better-balanced training data — 14.7B unique tokens across 5 sources vs 2.04B across 3, mixed with a staged (Warmup-Stable-Decay) schedule instead of a flat blend. Both share the exact same tokenizer and the exact same SFT dataset, so the comparison below is apples-to-apples.",
  },
  {
    q: "Is it actually better, or just bigger?",
    a: "Better on every shared metric measured — but not a clean sweep on every individual question. A direct, paired comparison on the identical 1,059 held-out questions found real regressions alongside the improvements (see the comparison table below). Bigger and better on net does not mean strictly better on every question — that nuance is documented, not glossed over.",
  },
  {
    q: "Should I trust its answers?",
    a: "For open-book tasks — give it a contract excerpt and ask about it — yes, that's the reliable mode (CUAD F1 0.761, LEDGAR 77.4%). For closed-book questions with no source text, no — it still confidently fabricates specifics on unfamiliar prompts, a hard capacity limitation at this parameter count, not a bug. Never use its output as legal, financial, or factual advice.",
  },
];

export default function Page() {
  return (
    <main style={{ position: "relative", zIndex: 2 }}>
      <Nav
        variant="500m"
        hfUrl={HF_URL}
        otherHref="/125m"
        otherLabel="Try the 125M model →"
        links={[
          { id: "chat", label: "Chat", small: false },
          { id: "compare", label: "vs. 125M", small: false },
          { id: "arch", label: "Architecture", small: true },
          { id: "about", label: "About", small: false },
        ]}
      />
      <Hero />

      <Section n="01" eyebrow="Chat" title="Ask it a question">
        <p style={lead}>
          Fine-tuned on the <b style={{ color: "var(--ink-soft)", fontWeight: 500 }}>exact same 21,186-pair
          SFT dataset</b> as the 125M model — CUAD contract clause extraction, LEDGAR clause classification,
          and distilled legal/financial Q&amp;A — so any difference in behavior below comes from the bigger,
          better-pretrained base, not different fine-tuning data. Runs entirely in your browser via
          WebAssembly; expect a larger download and slower generation than the 125M model.
        </p>
        <div style={{ marginTop: "1.5rem", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "1px", background: "var(--line)", border: "1px solid var(--line)", borderRadius: 5, overflow: "hidden", marginBottom: "1.75rem" }}>
          {SFT_STATS.map((s) => (
            <div key={s.k} style={{ background: "var(--paper-2)", padding: "1rem 1.1rem" }}>
              <div className="section-num" style={{ marginBottom: "0.35rem" }}>{s.k}</div>
              <div className="stat-num" style={{ fontSize: "1.15rem", color: "var(--ink)" }}>{s.v}</div>
              <div className="mono" style={{ fontSize: "0.68rem", color: "var(--faint)", marginTop: "0.2rem" }}>{s.note}</div>
            </div>
          ))}
        </div>
        <Chat
          variant="500m"
          presets={CHAT_PRESETS}
          hasServerMode={false}
          caveat="A 528.5M fine-tuned model. Meaningfully better than the 125M model, but still not reliable closed-book — see the comparison below. Not legal or financial advice."
        />
      </Section>

      <Section n="02" eyebrow="vs. 125M" title="Bigger, and honestly measured">
        <p style={lead}>
          Both models answering the <b style={{ color: "var(--ink-soft)", fontWeight: 500 }}>identical
          1,059 held-out questions</b>, scored the same way — extraction/classification against ground
          truth, general Q&amp;A judged by an independent local Meta-Llama-3.1-70B-Instruct.
        </p>
        <div className="paper-card" style={{ marginTop: "1.75rem", overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem", minWidth: 480 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--line)" }}>
                <th style={th}>Task</th>
                <th style={{ ...th, textAlign: "right" }}>125M</th>
                <th style={{ ...th, textAlign: "right" }}>500M</th>
              </tr>
            </thead>
            <tbody>
              {COMPARISON.map((c) => (
                <tr key={c.task} style={{ borderBottom: "1px solid var(--line)" }}>
                  <td style={td}>{c.task}</td>
                  <td style={{ ...td, textAlign: "right", fontFamily: "var(--font-mono)", color: c.better === "125m" ? "var(--green)" : "var(--faint)" }}>{c.v125}</td>
                  <td style={{ ...td, textAlign: "right", fontFamily: "var(--font-mono)", color: c.better === "500m" ? "var(--green)" : "var(--faint)" }}>{c.v500}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p style={{ ...lead, marginTop: "1.25rem", fontSize: "0.92rem" }}>
          The 500M model wins net in every category, but <b style={{ color: "var(--ink-soft)", fontWeight: 500 }}>not
          as a clean sweep</b> — e.g. on CUAD, 46 questions flipped from wrong to right, but 25 flipped the
          other way. One concrete regression: asked what changed conditions a defendant cited, the 125M model
          correctly listed them from the source text; the 500M model gave a vaguer, legal-sounding but
          non-responsive answer. Bigger and better on net does not mean strictly better on every question.
        </p>

        <h3 style={{ ...h3style, marginTop: "2.5rem" }}>Held-out perplexity, per source</h3>
        <p style={{ ...lead, marginBottom: "1.5rem" }}>
          The pretraining base model, not the fine-tuned one — measured on shared domains from the 125M
          model's own validation set, so the improvement isn't a val-set-composition artifact.
        </p>
        <div className="paper-card" style={{ padding: "1.5rem 1.25rem" }}>
          <PerSourcePplBars data={PERSOURCE_PPL} />
        </div>
      </Section>

      <Section n="03" eyebrow="The numbers" title="A bigger model, honest accounting">
        <div style={grid3}>
          {NUMBERS.map((x) => (
            <div key={x.k} style={numCell}>
              <div className="section-num" style={{ marginBottom: "0.5rem" }}>{x.k}</div>
              <div className="stat-num" style={{ fontSize: "1.9rem", color: "var(--ink)" }}>{x.v}</div>
              <div className="mono" style={{ fontSize: "0.74rem", color: "var(--faint)", marginTop: "0.25rem" }}>{x.note}</div>
            </div>
          ))}
        </div>
        <p style={{ ...lead, marginTop: "1.5rem", fontSize: "0.88rem" }}>
          <b style={{ color: "var(--ink-soft)", fontWeight: 500 }}>Honesty note on the aggregate perplexity:</b>{" "}
          it looks like a big win partly because one bulk source (SEC-EDGAR) is ~61% of the val set by
          window count and is inherently easier to predict (repetitive, formulaic filing text) than case
          law or general web text. The per-source numbers above — where this model beats the 125M model on
          all three shared domains — are the trustworthy signal, not the aggregate alone.
        </p>
      </Section>

      <Section n="04" eyebrow="Architecture" title="Same recipe, scaled up">
        <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", gap: "2.5rem", alignItems: "start" }}>
          <dl style={{ margin: 0, display: "grid", gap: 0 }}>
            {ARCH.map((a, i) => (
              <div key={a.k} style={{ display: "flex", justifyContent: "space-between", gap: "1rem", padding: "0.7rem 0", borderTop: i === 0 ? "none" : "1px solid var(--line)" }}>
                <dt style={{ color: "var(--muted)" }}>{a.k}</dt>
                <dd className="mono" style={{ margin: 0, color: "var(--ink)", fontSize: "0.85rem", textAlign: "right" }}>{a.v}</dd>
              </div>
            ))}
          </dl>
          <LayerStack />
        </div>
        <p className="mono" style={{ marginTop: "1.5rem", fontSize: "0.72rem", color: "var(--faint)" }}>
          {INFRA_NOTE}
        </p>
      </Section>

      <Section n="05" eyebrow="The corpus" title="Five sources, staged — not flat">
        <p style={lead}>
          A flat/proportional mix of these five sources was tried first and measurably diluted the original
          case-law/SEC/fineweb-edu identity down to ~13% of training data — fineweb-edu perplexity got{" "}
          <i>worse</i> (22.82 vs 20.73) as a direct result. The fix, adopted from MiniCPM's and SmolLM2's
          published recipes: a <b style={{ color: "var(--ink-soft)", fontWeight: 500 }}>staged (Warmup-Stable-Decay)
          schedule</b> — broad, balanced sampling for 85% of training, then a decay phase for the last 15%
          that upweights the curated core specifically. The donut below shows the time-averaged sampling
          weight across both phases.
        </p>
        <div className="paper-card" style={{ marginTop: "1.75rem", padding: "2rem" }}>
          <DonutMix mix={MIX} centerValue="14.7B" centerLabel="TOKENS" />
        </div>
      </Section>

      <Section n="06" eyebrow="Caveats" title="What this is, and is not">
        <div style={{ display: "grid", gap: "1.1rem", maxWidth: "46rem" }}>
          <Caveat>
            <b>Open-book contract tasks are the trustworthy mode</b> — give it an excerpt, and extraction/classification
            is measurably better than the 125M model across the board.
          </Caveat>
          <Caveat>
            <b>Closed-book general Q&amp;A improved but is still fundamentally limited.</b> A 528.5M-parameter
            model can store at most ~2 bits of knowledge per parameter (Allen-Zhu &amp; Li, "Physics of
            Language Models: Knowledge Capacity Scaling Laws") — a hard capacity ceiling, not a training defect.
          </Caveat>
          <Caveat>
            <b>Confident fabrication is real and was directly observed</b>, not hypothetical — invented specifics
            contradicting the real source, and the vague non-answer regression cited above. Never rely on its
            output as legal, financial, or factual advice.
          </Caveat>
        </div>
      </Section>

      <Section n="07" eyebrow="About" title="Who built this, and why">
        <div style={{ display: "grid", gap: "1.1rem", maxWidth: "52ch" }}>
          <p style={lead}>
            <strong style={{ color: "var(--ink)", fontWeight: 500 }}>{PERSON_NAME}</strong> — a{" "}
            {JOB_TITLE_LINE} — built this as the scaled-up successor to{" "}
            <a href="/125m" className="link-underline" style={{ color: "var(--green)" }}>legal-slm-125M ↗</a>,
            to test whether more parameters and more (better-balanced) legal/financial data would produce a
            genuinely stronger model, not just a bigger one. Every number on this page — including the ones
            that don't flatter the 500M model — comes from a real training or evaluation run.
          </p>
          <div style={{ display: "flex", gap: "1.25rem", flexWrap: "wrap", fontSize: "0.9rem", marginTop: "0.25rem" }}>
            <a href={GITHUB_URL} target="_blank" rel="noopener" className="link-underline" style={{ color: "var(--green)" }}>GitHub ↗</a>
            <a href={HF_PROFILE_URL} target="_blank" rel="noopener" className="link-underline" style={{ color: "var(--green)" }}>Hugging Face ↗</a>
            <a href={HF_URL} target="_blank" rel="noopener" className="link-underline">Base model ↗</a>
            <a href={HF_SFT_URL} target="_blank" rel="noopener" className="link-underline">SFT model ↗</a>
            <a href={HF_ONNX_URL} target="_blank" rel="noopener" className="link-underline">ONNX (browser) ↗</a>
          </div>
        </div>
      </Section>

      <Section n="08" eyebrow="FAQ" title="Questions worth answering honestly">
        <div style={{ display: "grid", gap: "1.5rem", maxWidth: "56rem" }}>
          {FAQS.map((f) => (
            <div key={f.q} style={{ display: "grid", gap: "0.4rem" }}>
              <h3 style={{ margin: 0, fontFamily: "var(--font-serif)", fontWeight: 500, fontSize: "1.15rem", color: "var(--ink)" }}>
                {f.q}
              </h3>
              <p style={{ margin: 0, color: "var(--muted)", lineHeight: 1.65, maxWidth: "60ch" }}>{f.a}</p>
            </div>
          ))}
        </div>
      </Section>

      <Footer />
    </main>
  );
}

/* ---------------- sections ---------------- */

function Hero() {
  return (
    <header id="top" style={{ position: "relative", overflow: "hidden" }}>
      <div className="wrap" style={{ paddingTop: "clamp(3.5rem, 9vw, 7rem)", paddingBottom: "clamp(3rem, 7vw, 5.5rem)" }}>
        <div className="rise">
          <div className="eyebrow" style={{ marginBottom: "1.4rem" }}>A 528.5-million-parameter base language model</div>
          <h1 className="display" style={{ fontSize: "clamp(2.6rem, 7vw, 5rem)", maxWidth: "22ch" }}>
            The 125M model&apos;s scaled-up, more honest successor.
          </h1>
          <p style={{ marginTop: "1.75rem", maxWidth: "46ch", fontSize: "1.1rem", color: "var(--muted)", lineHeight: 1.6 }}>
            Trained from a random initialization on <b style={{ color: "var(--ink-soft)", fontWeight: 500 }}>14.7&nbsp;billion tokens</b> across
            five sources of US case law, SEC filings, and educational web text — with a staged
            data schedule, not a flat blend.
          </p>
          <p className="mono" style={{ marginTop: "0.9rem", fontSize: "0.82rem", color: "var(--faint)", letterSpacing: "0.01em" }}>
            Built end-to-end by{" "}
            <a href={GITHUB_URL} target="_blank" rel="noopener" className="link-underline" style={{ color: "var(--muted)" }}>
              {PERSON_NAME}
            </a>{" "}
            — {JOB_TITLE_LINE}
          </p>
          <div style={{ marginTop: "2.25rem", display: "flex", gap: "0.9rem", flexWrap: "wrap", alignItems: "center" }}>
            <a href="#chat" className="btn-primary" style={{ display: "inline-block" }}>Chat with it ↓</a>
            <a href="#compare" className="btn-secondary" style={{ display: "inline-block" }}>See the comparison vs. 125M →</a>
          </div>
        </div>

        <div style={{ marginTop: "clamp(3rem, 7vw, 5rem)", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "1px", background: "var(--line)", border: "1px solid var(--line)", borderRadius: 5, overflow: "hidden" }}>
          {HERO_STATS.map((s) => (
            <div key={s.label} style={{ background: "var(--paper-2)", padding: "1.4rem 1.25rem" }}>
              <div className="stat-num" style={{ fontSize: "2rem", color: "var(--ink)" }}>{s.value}</div>
              <div className="mono" style={{ fontSize: "0.72rem", color: "var(--faint)", marginTop: "0.3rem", letterSpacing: "0.03em" }}>{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </header>
  );
}

function Section({ n, eyebrow, title, children }: { n: string; eyebrow: string; title: string; children: React.ReactNode }) {
  const anchor =
    eyebrow === "Chat" ? "chat" :
    eyebrow === "vs. 125M" ? "compare" :
    eyebrow === "Architecture" ? "arch" :
    eyebrow === "About" ? "about" :
    eyebrow === "FAQ" ? "faq" :
    undefined;
  return (
    <section id={anchor} style={{ borderTop: "1px solid var(--line)" }}>
      <div className="wrap" style={{ paddingTop: "clamp(3rem, 7vw, 5.5rem)", paddingBottom: "clamp(3rem, 7vw, 5.5rem)" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: "1.1rem", marginBottom: "1.5rem" }}>
          <span className="section-num">{n}</span>
          <div className="rule-brass" style={{ transform: "translateY(-4px)" }} />
          <span className="eyebrow">{eyebrow}</span>
        </div>
        <h2 className="display" style={{ fontSize: "clamp(1.9rem, 4.5vw, 3rem)", marginBottom: "0.5rem", maxWidth: "20ch" }}>{title}</h2>
        {children}
      </div>
    </section>
  );
}

function LayerStack() {
  return (
    <div className="paper-card" style={{ padding: "1.5rem", display: "flex", flexDirection: "column", gap: "0.9rem" }}>
      <Tag>tokens → 16,384 BPE embedding</Tag>
      <div style={{ display: "grid", gap: "3.5px" }}>
        {Array.from({ length: 24 }).map((_, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
            <span className="mono" style={{ fontSize: "0.58rem", color: "var(--faint)", width: 18 }}>{String(i + 1).padStart(2, "0")}</span>
            <div style={{ flex: 1, height: 10, borderRadius: 2, background: "var(--paper-3)", border: "1px solid var(--line-2)", position: "relative", overflow: "hidden" }}>
              <div style={{ position: "absolute", inset: 0, background: `linear-gradient(90deg, var(--green) ${9 + i}%, transparent ${9 + i}%)`, opacity: 0.16 }} />
            </div>
          </div>
        ))}
      </div>
      <Tag>RMSNorm → tied LM head → logits</Tag>
      <div className="mono" style={{ fontSize: "0.68rem", color: "var(--faint)", textAlign: "center", marginTop: "0.2rem" }}>
        24 decoder blocks · RoPE · SwiGLU
      </div>
    </div>
  );
}

function Tag({ children }: { children: React.ReactNode }) {
  return (
    <div className="mono" style={{ fontSize: "0.68rem", color: "var(--muted)", textAlign: "center", padding: "0.5rem", background: "var(--paper-3)", border: "1px solid var(--line)", borderRadius: 3 }}>
      {children}
    </div>
  );
}

function Caveat({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", gap: "0.9rem", alignItems: "flex-start" }}>
      <span style={{ color: "var(--brass)", fontFamily: "var(--font-serif)", fontSize: "1.4rem", lineHeight: 1, transform: "translateY(2px)" }}>§</span>
      <p style={{ margin: 0, color: "var(--ink-soft)", lineHeight: 1.6 }}>{children}</p>
    </div>
  );
}

function Footer() {
  return (
    <footer style={{ borderTop: "1px solid var(--line)", background: "var(--paper-3)" }}>
      <div className="wrap" style={{ padding: "2.5rem 1.75rem", display: "flex", flexWrap: "wrap", gap: "1.5rem", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div className="mono" style={{ fontSize: "0.8rem", letterSpacing: "0.14em", color: "var(--ink)" }}>
            LEGAL·SLM·<span style={{ color: "var(--green)" }}>500</span>
          </div>
          <p style={{ margin: "0.5rem 0 0", fontSize: "0.82rem", color: "var(--faint)", maxWidth: "42ch" }}>
            Built end-to-end by <strong style={{ color: "var(--muted)", fontWeight: 500 }}>{PERSON_NAME}</strong> —{" "}
            {JOB_TITLE_LINE}. Weights on Hugging Face · in-browser inference, no server required.
          </p>
        </div>
        <div style={{ display: "flex", gap: "1.5rem", fontSize: "0.85rem", flexWrap: "wrap" }}>
          <a href="/125m" className="link-underline">Try the 125M model →</a>
          <a href={HF_URL} target="_blank" rel="noopener" className="link-underline">Model ↗</a>
          <a href={GITHUB_URL} target="_blank" rel="noopener" className="link-underline">GitHub ↗</a>
          <a href={HF_PROFILE_URL} target="_blank" rel="noopener" className="link-underline">Hugging Face ↗</a>
          <a href="#top" className="link-underline">Back to top ↑</a>
        </div>
      </div>
    </footer>
  );
}

/* ---------------- shared styles ---------------- */
const lead: React.CSSProperties = { maxWidth: "56ch", fontSize: "1.08rem", color: "var(--muted)", lineHeight: 1.65 };
const grid3: React.CSSProperties = { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "1px", background: "var(--line)", border: "1px solid var(--line)", borderRadius: 5, overflow: "hidden" };
const numCell: React.CSSProperties = { background: "var(--paper-2)", padding: "1.6rem 1.5rem" };
const th: React.CSSProperties = { textAlign: "left", padding: "0.85rem 1.1rem", fontSize: "0.72rem", letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--faint)" };
const td: React.CSSProperties = { padding: "0.75rem 1.1rem", color: "var(--ink)" };
const h3style: React.CSSProperties = { fontFamily: "var(--font-serif)", fontWeight: 500, fontSize: "1.25rem", color: "var(--ink)", margin: 0 };
