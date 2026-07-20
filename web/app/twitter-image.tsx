import { ImageResponse } from "next/og";
import { HERO_STATS } from "@/app/lib/model";
import { PERSON_NAME } from "@/app/lib/seo";

// Next.js's route-segment-config parser needs these exports defined
// directly in this file -- re-exporting them from opengraph-image.tsx
// (via `export { runtime, size, ... } from "./opengraph-image"`) fails
// the Turbopack build ("mustn't be reexported"), confirmed live.
export const runtime = "edge";
export const alt = "legal-slm-125M — a 125M-parameter legal & financial LLM built from scratch by Deependra Verma";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function TwitterImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "72px 80px",
          background: "#0a0b12",
          color: "#eef0f7",
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", fontSize: 26, letterSpacing: 4, color: "#8b8ff8", textTransform: "uppercase" }}>
            LEGAL·SLM·125M
          </div>
          <div style={{ display: "flex", marginTop: 28, fontSize: 60, lineHeight: 1.12, maxWidth: 980, color: "#eef0f7" }}>
            A 125M-parameter legal &amp; financial language model, built from scratch.
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div style={{ display: "flex", gap: 44 }}>
            {HERO_STATS.map((s) => (
              <div key={s.label} style={{ display: "flex", flexDirection: "column" }}>
                <div style={{ display: "flex", fontSize: 40, color: "#eef0f7" }}>{s.value}</div>
                <div style={{ display: "flex", fontSize: 18, color: "#696d82", letterSpacing: 1 }}>{s.label}</div>
              </div>
            ))}
          </div>
          <div style={{ display: "flex", fontSize: 24, color: "#9195ab" }}>
            Built end-to-end by {PERSON_NAME} — Generative AI Researcher &amp; AI Engineer
          </div>
        </div>
      </div>
    ),
    { ...size },
  );
}
