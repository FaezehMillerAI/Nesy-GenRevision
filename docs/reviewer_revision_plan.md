# Reviewer-Driven AAAI Revision Plan

This tracker summarizes how the implementation addresses the main weaknesses in
the earlier manuscript and what evidence is still needed for a solid AAAI
submission.

## Addressed In Code

| Reviewer concern | Current implementation |
| --- | --- |
| Reproducibility unclear | Installable package, tests, scripts, and Colab notebooks |
| Real PrimeKG use unclear | Dataverse loader plus reusable radiology PrimeKG cache |
| Entity linking not validated | Entity-linking validation bundle with raw/filtered links, coverage, frequencies, and manual audit sample |
| Graph reasoning too abstract | PrimeKG subgraph builder, LTN-style clause scores, and candidate audit CSVs |
| No robustness analysis | Sensitivity tools for dropped/swapped links and graph coverage |
| No latency/accounting | Profiling utilities and candidate audit outputs |
| Generation not evaluated enough | Lexical metrics, clinical proxy metrics, official metric adapters, leakage audit |
| Hallucination claims too broad | Claims reframed as unsupported entity/clinical-label proxy measurements |

## Proposed AAAI Method Framing

Nesy-Gen should be presented as an improved neuro-symbolic generation framework,
not as a new backbone architecture. The image-to-report generator is a component;
the contribution is the graph-grounded verification and selection layer:

1. Vision-T5 candidate report generation.
2. Retrieval of leakage-safe clinical evidence.
3. PrimeKG entity grounding and radiology subgraph construction.
4. LTN-style fuzzy graph verification.
5. Consistency-gated final report selection.
6. Optional graph-aware training and soft graph-constrained decoding.

## Remaining Evidence Needed

| Evidence | Why it matters | Status |
| --- | --- | --- |
| Full IU-Xray standard vs graph-constrained run | Main proof that the method helps beyond smoke runs | Needed |
| Full MIMIC-CXR run | External dataset evidence | Needed |
| Retrieval-only baseline | Shows graph method is not just retrieval | Needed |
| Vision-T5 without graph gate | Isolates graph contribution | Needed |
| Official CheXbert/CheXpert results | Clinical label validity | Needed |
| Official RadGraph results | Entity/relation factuality | Needed |
| Manual entity-linking audit | Validates the graph grounding input | Needed |
| Template repetition/diversity table | Guards against artificially high BLEU from repeated reports | Needed |
| Candidate-level qualitative examples | Makes ante-hoc verification visible to reviewers | Needed |

## Recommended Claim Boundary

Use:

> Nesy-Gen improves graph-grounded explainability and reduces unsupported
> entity generation under reference-anchored clinical/entity metrics.

Avoid:

> Nesy-Gen eliminates hallucination.

The latter requires radiologist adjudication or a stronger factuality benchmark
than automated reference proxies alone.

## Main Experiments To Run

For both IU-Xray and MIMIC-CXR:

1. Retrieval baseline.
2. Vision-T5 generator without PrimeKG verification.
3. RAG + Vision-T5 without graph-constrained decoding.
4. RAG + PrimeKG/LTN consistency gate.
5. RAG + PrimeKG/LTN gate + graph-constrained decoding.
6. Optional graph-aware training ablation.

The final tables should include lexical metrics, official clinical metrics,
entity-level unsupported content proxies, graph satisfaction, diversity, latency,
and qualitative examples.
