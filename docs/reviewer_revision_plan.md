# Reviewer-Driven Revision Plan

## Methodology Fixes Implemented in Code

- Reproducible package structure with installable modules, tests, scripts, and a
  Colab notebook.
- PrimeKG loader with graph coverage reporting for missing entities.
- Swappable entity-linking interface so scispaCy can be compared with RadGraph,
  CheXbert, RaTEScore, or a deterministic lexical crosswalk.
- Temporal Steiner-style patient subgraph builder using relation and temporal
  weights from the manuscript.
- LTN-style fuzzy clause satisfaction for biological/temporal validity,
  finding-diagnosis connectivity, and `located_in` type constraints.
- Consistency gate over graph validity, visual grounding, NLI entailment, and
  hallucination risk.
- Sensitivity analysis for dropped and swapped entity links.
- Latency and model-size utilities for computational overhead reporting.

## Manuscript Revisions Needed for EACL

- Correct title typo: `Graph-Verfied` -> `Graph-Verified`.
- Narrow unsupported "state-of-the-art" wording unless direct comparisons with
  R2GenKG, MAIRA-2, CXR-LLaVA, CXR-Mate, and GraphRAG-style baselines are added.
- Add a direct factuality section: hallucination rate, RadGraph/RaTEScore-style
  metrics, and radiologist review if feasible.
- Add entity-linking validation and error-propagation sensitivity tables.
- Add inference latency, GPU memory, parameter count, and per-module cost.
- Reframe the method as graph-verified generation related to GraphRAG, then
  clarify what is ante-hoc verification versus retrieval-only augmentation.
- Replace non-standard `[?]` and `[!]` token labels in Figure 2 with clearer
  labels such as `FLAGGED` and `REJECTED/UNCERTAIN`.

