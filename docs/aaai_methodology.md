# AAAI Methodology Notes

This document is the recommended paper-facing description of the improved
Nesy-Gen implementation.

## Proposed Method

Nesy-Gen is a graph-grounded report generation framework. The image-to-text
generator is treated as a backbone, while the contribution is the
neuro-symbolic layer around it:

1. **Visual report generator:** a frozen visual encoder produces spatial image
   tokens. A T5 decoder generates candidate reports from those image tokens.
2. **Retrieval evidence:** a leakage-safe retrieval module retrieves similar
   training reports using inference-available fields, not the test reference.
3. **PrimeKG entity grounding:** generated and retrieved report concepts are
   linked to PrimeKG node identifiers with negation awareness.
4. **Radiology PrimeKG subgraph:** a reusable radiology-focused PrimeKG cache is
   built once from report entities and radiology seed terms.
5. **LTN-style consistency audit:** each candidate is scored using fuzzy graph
   clauses for biological/temporal support, finding-diagnosis connectivity, and
   anatomy/finding type compatibility.
6. **Consistency gate:** retrieval evidence, graph consistency, and entity-level
   support are combined to select the final report.

The implementation supports optional graph-aware token training and optional
soft graph-constrained decoding. These are ablation variables, not mandatory
assumptions.

## Main Scientific Claims

Use these claims only when supported by the final full-split tables:

- Nesy-Gen improves explainability by attaching generated report content to
  PrimeKG-linked concepts and graph paths.
- Nesy-Gen provides ante-hoc verification: reports are generated as candidates,
  checked by PrimeKG/LTN constraints, and selected through a consistency gate
  before final output.
- Graph-constrained decoding and graph-aware training can reduce unsupported
  clinical entity generation relative to the same generator without graph
  constraints.
- The method exposes failure modes through entity-linking validation, coverage
  tables, graph satisfaction distributions, and candidate audit logs.

Avoid claiming that the method "solves hallucination" or is state-of-the-art
unless official metrics and direct comparisons support that conclusion.

## Recommended Main Experiments

Run the following for IU-Xray and MIMIC-CXR:

| Variant | Purpose |
| --- | --- |
| Retrieval baseline | Strong non-generative reference and leakage check |
| Vision-T5 generator | Measures backbone generation quality |
| RAG + generator | Tests retrieval evidence without graph verification |
| RAG + PrimeKG/LTN gate | Tests ante-hoc graph verification |
| Graph-constrained Nesy-Gen | Tests soft PrimeKG decoding plus graph gate |
| Graph-aware training ablation | Tests whether PrimeKG token weighting helps |

For the final paper, prefer the simplified Colab notebook for full runs and the
advanced notebook only for ablations.

## Required Evaluation

Report three groups of metrics.

### 1. Text Generation Quality

- BLEU-1/2/3/4
- ROUGE-L
- METEOR
- CIDEr
- prediction diversity diagnostics, especially repeated-template rate

Use official COCO-caption tooling for final tables where possible. The built-in
development metrics are useful for triage but should be labelled as lightweight
diagnostics.

### 2. Clinical Correctness And Unsupported Entity Proxies

- CheXbert or CheXpert label precision, recall, F1
- positive-label hallucination rate:

  \[
  H_{label} =
  \frac{|P^+_{pred} \setminus P^+_{ref}|}
       {\max(1, |P^+_{pred}|)}
  \]

  where \(P^+_{pred}\) and \(P^+_{ref}\) are positive clinical labels in the
  generated and reference reports.

- entity-level unsupported rate:

  \[
  H_{entity} =
  \frac{|E^+_{pred} \setminus E^+_{ref}|}
       {\max(1, |E^+_{pred}|)}
  \]

  where \(E^+\) denotes non-negated PrimeKG-linked clinical entities.

These are proxies for unsupported content. They should be described as
reference-anchored factuality proxies, not as definitive hallucination labels.

### 3. Graph And Explanation Quality

- mean PrimeKG/LTN graph satisfaction;
- clause-level satisfaction for biological/temporal support,
  finding-diagnosis support, and location/type compatibility;
- number of linked entities per report;
- positive vs negated entity counts;
- graph coverage of extracted concepts;
- qualitative examples showing accepted/rejected candidates and graph evidence.

## Entity Linking Validation

The entity-linking validation bundle should be included in the appendix or
supplement:

- raw links and filtered links;
- coverage by report;
- entity frequency table;
- condition-to-link coverage table;
- manual audit sample with text spans, node IDs, node names, node types, and
  negation flags.

Use the manual audit sample to report:

- mention detection precision;
- PrimeKG node-linking accuracy;
- negation accuracy;
- most common false positive and false negative entity categories.

This directly addresses reviewer concerns that graph reasoning quality depends
on extraction/linking quality.

## Defensible Design Choices

- **Frozen visual encoder:** reduces compute and overfitting risk on small
  radiology datasets; keeps the graph contribution isolated from backbone
  capacity changes.
- **ConvNeXt/Swin-style spatial features:** provide stronger visual tokens than
  a single global pooled feature while keeping the model manageable on Colab A100.
- **Soft constraints rather than hard templates:** preserves lexical variety and
  avoids forcing graph entities into every report.
- **Candidate audit CSVs:** make the selection process inspectable and allow
  reviewers to see whether improvements come from retrieval, generation, graph
  score, or gate behavior.
- **Radiology PrimeKG cache:** makes full PrimeKG usable at experiment time while
  remaining traceable to the original Dataverse files.

## Implementation Boundary

Legacy module names containing `r2gen_t5` are retained for checkpoint and script
compatibility. The proposed method should be referred to as **Nesy-Gen** or
**Vision-T5 Nesy-Gen**, not as R2Gen.
