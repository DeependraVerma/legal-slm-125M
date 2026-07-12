"use client";

import { useEffect, useState } from "react";
import ThemeToggle from "@/app/components/ThemeToggle";
import { HF_URL } from "@/app/lib/model";

const LINKS = [
  { id: "play", label: "Playground", small: false },
  { id: "chat", label: "Chat", small: false },
  { id: "arch", label: "Architecture", small: true },
];

export default function Nav() {
  const [active, setActive] = useState<string>("");

  useEffect(() => {
    const els = LINKS.map((l) => document.getElementById(l.id)).filter(
      (el): el is HTMLElement => !!el,
    );
    // Active = the section crossing a thin band ~45% down the viewport.
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) if (e.isIntersecting) setActive(e.target.id);
      },
      { rootMargin: "-45% 0px -50% 0px", threshold: 0 },
    );
    els.forEach((el) => obs.observe(el));
    return () => obs.disconnect();
  }, []);

  return (
    <nav style={{ position: "sticky", top: 0, zIndex: 20, backdropFilter: "saturate(1.2) blur(8px)", background: "var(--nav-bg)", borderBottom: "1px solid var(--line)" }}>
      <div className="wrap" style={{ height: 60, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <a href="#top" className="mono" style={{ fontSize: "0.8rem", letterSpacing: "0.14em", color: "var(--ink)" }}>
          LEGAL·SLM·<span style={{ color: "var(--green)" }}>125</span>
        </a>
        <div style={{ display: "flex", gap: "1.4rem", alignItems: "center", fontSize: "0.86rem" }}>
          {LINKS.map((l) => (
            <a
              key={l.id}
              href={`#${l.id}`}
              className={`link-underline${l.small ? " hide-sm" : ""}`}
              style={active === l.id ? { color: "var(--green)" } : undefined}
            >
              {l.label}
            </a>
          ))}
          <a href={HF_URL} target="_blank" rel="noopener" className="link-underline hide-sm">
            Hugging Face ↗
          </a>
          <ThemeToggle />
        </div>
      </div>
    </nav>
  );
}
