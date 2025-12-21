# RAG System Evaluation Report

## Overview

This report summarizes a series of controlled experiments conducted using the `evaluate_rag_openai.py` evaluation script. The goal was to assess the impact of prompt design, document splitting, reranking, and hypothetical document embeddings on overall RAG answer quality, as measured by an LLM-as-a-judge framework. Test questions and their corresponding expected answers are stored in the data/test/ directory. All original PDF documents used in the RAG experiments are preserved in the data/pdf/ directory

The evaluation uses a **percentage-based scoring system (0–100%)**, where each answer is judged against a curated expected answer along three dimensions:

* **Factual correctness**
* **Completeness**
* **Faithfulness to retrieved context**

The evaluator LLM outputs both a numeric score and a brief justification, enabling quantitative benchmarking and qualitative error analysis. This setup supports regression testing, ablation studies, and CI integration.

All evaluation outputs, including per-question scores, justifications, and aggregated metrics, are automatically saved in the eval/ directory for traceability, regression analysis, and CI integration.

---

## 0. Baseline Performance

* **Baseline overall average score:** **66.25**

This baseline reflects the original RAG configuration prior to prompt, chunking, or retrieval enhancements.

---

## 1. Prompt Change Experiment

### Change Description

The generation prompt was replaced with a highly restrictive, context-only prompt enforcing strict grounding rules:

* Answers must rely **exclusively** on provided context
* Explicit fallback response when context is insufficient
* No correction of context errors
* Structured output with a direct answer and bullet-point support

### Observed Impact

* **Overall average score:** **54.5** (↓ from 66.25)

### Analysis

The stricter prompt significantly reduced scores. A large fraction of responses correctly—but frequently—returned:

> *“The provided context does not contain the answer.”*

While this behavior improves **faithfulness**, it negatively impacts **completeness** and overall scoring under the current evaluation rubric. This suggests a misalignment between:

* The evaluation criteria (which reward completeness), and
* The prompt’s conservative refusal policy.

### Key Insight

Strict factual grounding increases safety and faithfulness but requires:

* Better retrieval recall, or
* Updated evaluation expectations that explicitly reward correct abstention.

---

## 2. Sentence-Based Chunking with Overlap

### Change Description

Paragraph-based splitting was replaced with sentence-level sliding-window chunking with overlap.

**Configuration:**

```python
cleaner = PDFCleaner(
    min_block_size=300,
    max_block_size=1000,
    overlap_sentences=1
)
```

**Mechanism:**

* Text is split into sentences
* Sentences are aggregated until size constraints are met
* Overlapping windows preserve local context continuity

### Observed Impact

* **Overall average score:** **64.125**

### Analysis

This approach improved contextual coherence and reduced fragmentary retrieval results compared to paragraph splitting. Although still slightly below the baseline, the results indicate improved recall and answer grounding, especially for multi-sentence facts.

---

## 3. Reranker Integration

### Change Description

A reranking stage was added after initial vector retrieval:

* **Primary reranker:** `ms-marco-MiniLM-L-6-v2` (cross-encoder)
* **Fallback:** embedding cosine similarity
* **Selection:** top-10 documents passed to generation from top-15 retrieved

### Observed Impact

* **Overall average score:** **68.0**

### Analysis

Reranking significantly improved retrieval precision by reducing contextual noise and prioritizing semantically relevant passages. This led to more focused, faithful answers and surpassed the original baseline.

### Key Benefit

* Improved answer grounding without increasing prompt complexity
* Robustness ensured via fallback mechanism

---

## 4. Hypothetical Document Embeddings (HyDE)

### Change Description

A hypothetical document generation prompt was introduced to enrich query embeddings prior to retrieval.

**Key constraints:**

* 1–3 sentences only
* No meta-commentary
* No mention of hypothetical nature

### Observed Impact

* **Overall average score:** **68.875** (best result)

### Analysis

HyDE improved recall by expanding sparse or underspecified queries into semantically richer representations. This enhancement worked particularly well in combination with reranking, leading to the highest observed overall score.

---

## Summary of Results

| Configuration                      | Avg. Score |
| ---------------------------------- | ---------- |
| Baseline                           | 66.25      |
| Strict Context-Only Prompt         | 54.5       |
| Sentence-Based Chunking            | 64.125     |
| + Reranker                         | 68.0       |
| + Hypothetical Document Embeddings | **68.875** |

---

## Key Conclusions

1. **Prompt strictness alone can degrade performance** if retrieval recall is insufficient or evaluation does not reward abstention.
2. **Retrieval quality improvements (chunking, reranking, HyDE)** have a stronger positive impact than prompt changes.
3. **Reranking and HyDE together outperform the baseline**, indicating that upstream retrieval enhancements are the highest-leverage interventions.