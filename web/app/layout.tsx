import type { Metadata } from "next";
import { Fraunces, Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

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

export const metadata: Metadata = {
  title: "legal-slm-125 · a 125M legal & financial language model",
  description:
    "A 125-million-parameter base language model trained from scratch on 2.04 billion tokens of US case law, SEC filings and educational web text. Watch it complete legal and financial passages, live.",
  openGraph: {
    title: "legal-slm-125",
    description:
      "A 125M base language model for legal & financial text. Held-out perplexity 9.13. Trained from nothing on 8×H100. Try it live.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${fraunces.variable} ${inter.variable} ${jetbrains.variable}`}
      >
        {children}
      </body>
    </html>
  );
}
