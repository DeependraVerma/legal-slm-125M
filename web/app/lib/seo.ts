// SEO / AEO (answer-engine) constants and structured-data builders.
// Every fact referenced here comes straight from README.md — nothing here
// is invented or approximated for search-engine effect.

export const SITE_URL = "https://deependraverma-ai-legal-slm-125-m.vercel.app";
export const SITE_NAME = "legal-slm-125M";

export const PERSON_NAME = "Deependra Verma";
export const PERSON_URL = "https://github.com/DeependraVerma";
export const GITHUB_URL = "https://github.com/DeependraVerma";
export const GITHUB_REPO_URL = "https://github.com/DeependraVerma/legal-slm-125M";
export const HF_PROFILE_URL = "https://huggingface.co/DeependraVerma";

// The professional titles Deependra Verma wants associated with their name.
// Repeated as a combined string (for natural on-page copy / meta description)
// and as a list (for Person.jobTitle in the JSON-LD, which accepts repeated
// property values).
export const JOB_TITLES = ["Generative AI Researcher", "AI Engineer", "AI Team Lead"] as const;
export const JOB_TITLE_LINE = "Generative AI Researcher, AI Engineer, and AI Team Lead";

// Only profile links actually documented in this repository — no invented
// LinkedIn / X / personal-site URLs.
export const SAME_AS = [GITHUB_URL, HF_PROFILE_URL];

export const PROJECT_DESCRIPTION =
  "A 125.8-million-parameter Llama-style decoder language model, built entirely from scratch " +
  "for legal and financial English: data pipeline, byte-level BPE tokenizer, pretraining, " +
  "held-out evaluation, a live in-browser demo, and a supervised fine-tuned Q&A assistant.";

export const FAQS: { q: string; a: string }[] = [
  {
    q: "Who built legal-slm-125M?",
    a: "It was designed and built end-to-end by Deependra Verma — a Generative AI Researcher, AI Engineer, and AI Team Lead — covering the full pipeline from data cleaning and tokenizer training through pretraining, evaluation, and fine-tuning.",
  },
  {
    q: "What is legal-slm-125M?",
    a: "A 125.8-million-parameter Llama-style decoder language model trained from a random initialization on 2.04 billion unique tokens of US case law, SEC filings, and educational web text, reaching a held-out perplexity of 7.76 after two epochs of pretraining (38,890 steps).",
  },
  {
    q: "What's the difference between the base model and the fine-tuned (chat) model?",
    a: "The base model (slm-125m-base) only continues text — it was never shown a question-answer pair and cannot answer questions. The fine-tuned model (legal-slm-125m-sft) was further trained via supervised fine-tuning on a teacher-LLM-distilled, judge-filtered legal/financial Q&A dataset, so it can actually answer questions, though both models still fabricate specifics on unfamiliar prompts.",
  },
  {
    q: "Is the fine-tuned model production-ready or safe to rely on for legal or financial advice?",
    a: "No. It answers reasonably on prompts similar to its training distribution, but like any 125-million-parameter model fine-tuned on a few thousand examples, it confidently invents case names, citations, and figures on out-of-distribution or adversarial questions. This is a research and portfolio project, not a source of legal, financial, or factual advice.",
  },
  {
    q: "How was the model trained, and what did it cost?",
    a: "Pretraining ran for two epochs on a from-scratch cleaned and deduplicated corpus, and the fine-tuning stage distilled a Q&A dataset from a teacher LLM, filtered by an LLM-as-judge. The compute-heavy phases (pretraining, evaluation, fine-tuning) ran on the author's own on-prem GPUs rather than rented cloud instances, and the fine-tuning dataset was built entirely on free-tier APIs — real out-of-pocket cost for this build was $0.",
  },
];

export function personJsonLd() {
  return {
    "@type": "Person",
    "@id": `${SITE_URL}/#person`,
    name: PERSON_NAME,
    url: PERSON_URL,
    jobTitle: [...JOB_TITLES],
    description: `${PERSON_NAME} is a ${JOB_TITLE_LINE.toLowerCase()} who designed and built ${SITE_NAME} end-to-end — data pipeline, tokenizer, pretraining, evaluation, and fine-tuning.`,
    sameAs: SAME_AS,
    knowsAbout: [
      "Large Language Models",
      "Generative AI",
      "Language Model Pretraining",
      "Supervised Fine-Tuning",
      "Natural Language Processing",
      "Machine Learning Engineering",
    ],
  };
}

export function softwareJsonLd() {
  return {
    "@type": "SoftwareSourceCode",
    "@id": `${SITE_URL}/#software`,
    name: SITE_NAME,
    description: PROJECT_DESCRIPTION,
    codeRepository: GITHUB_REPO_URL,
    url: SITE_URL,
    programmingLanguage: ["Python", "TypeScript"],
    author: { "@id": `${SITE_URL}/#person` },
    creator: { "@id": `${SITE_URL}/#person` },
    keywords: [
      "large language model",
      "LLM pretraining",
      "legal AI",
      "financial NLP",
      "transformer",
      "supervised fine-tuning",
    ],
  };
}

export function faqJsonLd() {
  return {
    "@type": "FAQPage",
    "@id": `${SITE_URL}/#faq`,
    mainEntity: FAQS.map((f) => ({
      "@type": "Question",
      name: f.q,
      acceptedAnswer: { "@type": "Answer", text: f.a },
    })),
  };
}

export function siteJsonLdGraph() {
  return {
    "@context": "https://schema.org",
    "@graph": [personJsonLd(), softwareJsonLd(), faqJsonLd()],
  };
}
