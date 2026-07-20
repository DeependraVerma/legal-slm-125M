import type { Metadata } from "next";
import { Fraunces, Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { PERSON_NAME, SITE_NAME, SITE_URL, siteJsonLdGraph } from "@/app/lib/seo";

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-fraunces",
  display: "swap",
});
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});
const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono-jb",
  display: "swap",
});

const TITLE = `${PERSON_NAME} — legal-slm-125M, a 125M-parameter LLM built from scratch`;
const DESCRIPTION =
  "legal-slm-125M is a 125.8M-parameter legal & financial language model built end-to-end by " +
  "Deependra Verma — Generative AI Researcher, AI Engineer, and AI Team Lead — from a random " +
  "initialization through pretraining (held-out perplexity 7.76 on 2.04B tokens), evaluation, " +
  "and supervised fine-tuning into a live Q&A assistant.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: TITLE,
    template: `%s · ${SITE_NAME}`,
  },
  description: DESCRIPTION,
  keywords: [
    "Deependra Verma",
    "Deependra Verma Generative AI Researcher",
    "Deependra Verma AI Engineer",
    "Deependra Verma AI team lead",
    "legal-slm-125M",
    "legal language model",
    "LLM built from scratch",
    "small language model",
    "legal AI",
    "financial NLP",
    "language model pretraining",
    "supervised fine-tuning",
  ],
  authors: [{ name: PERSON_NAME, url: "https://github.com/DeependraVerma" }],
  creator: PERSON_NAME,
  publisher: PERSON_NAME,
  alternates: {
    canonical: "/",
  },
  openGraph: {
    title: TITLE,
    description: DESCRIPTION,
    url: SITE_URL,
    siteName: SITE_NAME,
    type: "website",
    locale: "en_US",
    images: [{ url: "/opengraph-image", width: 1200, height: 630, alt: TITLE }],
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
    images: ["/opengraph-image"],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: { index: true, follow: true },
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${fraunces.variable} ${inter.variable} ${jetbrains.variable}`}
      >
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem('theme');if(t!=='light'&&t!=='dark'){t=window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}document.documentElement.setAttribute('data-theme',t);}catch(e){}})();`,
          }}
        />
        <script
          type="application/ld+json"
          // eslint-disable-next-line react/no-danger
          dangerouslySetInnerHTML={{ __html: JSON.stringify(siteJsonLdGraph()) }}
        />
        {children}
      </body>
    </html>
  );
}
