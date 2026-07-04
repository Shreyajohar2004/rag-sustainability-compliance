# RAG for Regulatory Compliance: EU ESRS & ISSB Sustainability Standards

A Retrieval-Augmented Generation (RAG) system that answers questions on EU Sustainability Reporting Standards (ESRS) and global ISSB standards (IFRS S1/S2) — evaluated against a plain LLM to test whether retrieval actually earns its keep for compliance-grade Q&A.

## The problem

LLMs answer fluently but from fixed, dated training data — a real liability for regulatory questions where wording is precise, rules change over time, and an invented answer carries genuine compliance risk. This project builds and evaluates a RAG pipeline over ESRS/ISSB source text, including a regulation (the November 2025 "Quick Fix" amending the ESRS phase-in timetable) published *after* the base model's training cut-off — giving a clean, unambiguous test of whether retrieval adds knowledge the model cannot already have.

## What was built

Three pipelines were compared on an identical 20-query test set (simple-fact, deep-context, ambiguous plain-English, edge/refusal, and post-training queries):

- **No-RAG baseline** — the LLM (Llama-3.1-8b-instant via Groq) answers directly, no retrieval.
- **Baseline RAG** — fixed-size chunking (1,000 characters, 150 overlap), dense retrieval over `BAAI/bge-small-en-v1.5` embeddings in ChromaDB, plain prompt.
- **Enhanced RAG** — four techniques layered on top:
  - **Structure-aware chunking** — splits on the standards' own paragraph numbering rather than arbitrary character windows
  - **Query rewriting (HyDE)** — reformulates plain-English questions into standards terminology before retrieval
  - **Hybrid retrieval (BM25 + dense) with Reciprocal Rank Fusion** — combines exact keyword matching with semantic search
  - **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`) — jointly scores query-chunk pairs to sharpen the final top-k

Generation uses a grounded, citation-enforcing prompt that requires a bracketed source for every claim and an explicit refusal when the retrieved context doesn't support an answer.

## Results

| Arm | Precision@k | Recall@k | MRR | Correctness | Groundedness |
|---|---|---|---|---|---|
| No-RAG | — | — | — | 3.60 | 4.1 |
| Baseline RAG | 0.36 | 0.134 | 0.460 | 4.65 | 4.8 |
| Enhanced RAG | 0.30 | 0.188 | 0.535 | 4.80 | 4.9 |

**The clearest result:** on the six queries answerable only from the post-cutoff "Quick Fix" regulation, the no-RAG baseline scored **1.0/5** correctness — it simply didn't know the regulation existed — while both RAG arms scored **5.0/5**, correctly stating the precise relief provisions. This isolates RAG's value from anything the base model could already know.

Enhanced retrieval improved recall (+40%) and ranking quality (MRR +16%) over the baseline, at a modest cost in precision — an expected and explained trade-off: casting a wider net via hybrid fusion surfaces more relevant provisions and ranks the first hit higher, at the cost of admitting a few more borderline chunks into the top-k.

## Honest limitations

- **Judge saturation:** generation quality is scored by the same 8B model that generates the answers, and it's lenient — most arms cluster near 5/5, so generation metrics should be read as a ceiling check rather than a precise ranking. Retrieval metrics (precision, recall, MRR) are the more trustworthy signal.
- **Grounding can over-refuse:** on one deep-context query the enhanced arm declined to answer a well-covered topic because retrieval didn't surface a chunk it judged sufficient — a defensible failure mode for compliance (a confident wrong citation is worse than an honest "not found") but one that would need threshold tuning before deployment.
- **Not production-scale:** in-memory vector store rebuilt each session, added latency from the hybrid+rerank pipeline, and a fixed corpus snapshot that would need re-ingestion as regulation evolves.

## Why this matters

For compliance RAG specifically, retrieval quality and disciplined grounding matter more than generation fluency — and evaluation needs to look past saturated generation scores to the retrieval metrics and failure cases that actually distinguish a usable system from a plausible-sounding one.

## Stack

Python · Llama-3.1-8b-instant (Groq API) · ChromaDB · `sentence-transformers` (bge-small-en-v1.5) · BM25 · cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) · PyMuPDF

---
*Built as part of an MSc Business Analytics module (Generative AI and AI Applications), Warwick Business School.*
