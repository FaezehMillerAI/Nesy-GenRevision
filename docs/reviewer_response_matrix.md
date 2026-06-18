# Reviewer Response Matrix

This matrix maps the original reviewer and meta-reviewer concerns to concrete
implementation artifacts and remaining paper actions.

## Summary

The revision should be positioned as a major, evidence-driven extension of
Nesy-Gen. The strongest defensible claim is not "state-of-the-art report
generation"; it is **graph-grounded, ante-hoc verification for explainable
radiology report generation**, evaluated with stronger baselines, official
metrics, entity-linking validation, and ablations.

## Concern-To-Artifact Mapping

| Reviewer concern | Implementation response | Paper action |
| --- | --- | --- |
| Computational complexity and inference latency are unclear | Radiology PrimeKG cache, ego subgraph mode, latency utilities, candidate audit CSVs, `run_primekg_reasoning.py --latency-repeats` | Add runtime table with mean/p50/p95 latency, GPU, batch size, graph cache size, and parameter count |
| Sensitivity to PrimeKG quality and missing graph concepts | Graph coverage tables, linked/unlinked entity counts, raw vs filtered entity links, sensitivity analysis under dropped/swapped links | Add missing-concept behavior section: unsupported or unlinked entities are flagged, not silently accepted |
| Entity extraction and normalization under-validated | `scripts/validate_entity_linking.py` exports raw links, filtered links, coverage, frequencies, condition coverage, and manual audit sample | Report mention precision, node-linking accuracy, negation accuracy, and common failure categories from manual audit |
| Hallucination/factuality evidence is indirect | Entity-level unsupported-content proxy, CheXbert/CheXpert adapter, RadGraph adapter, official metric input export | Claim "unsupported entity reduction" and "graph-grounded explanation", not general hallucination elimination |
| Baselines are incomplete | Retrieval baseline, raw Vision-T5, RAG + PrimeKG gate, graph-constrained decoding, graph-aware training ablation, comparison script | Compare against retrieval and internal ablations; cite external SOTA carefully if direct runs are not feasible |
| RAG relationship unclear | RAG candidates are explicitly included; PrimeKG/LTN gate is positioned as verification beyond retrieval | Explain Nesy-Gen as RAG-compatible graph verification, not as a replacement for RAG |
| KG-centric metric may favor the method | Official CheXbert/CheXpert and RadGraph adapters are included; quick KG metrics labelled as diagnostics | Put official clinical metrics first in final tables, KG metrics as explanation/diagnostic evidence |
| Decoder/model size unclear | Experiment config writer records text model, visual backbone, frozen encoder, parameter/profiling hooks | Add model card-style table with backbone, decoder, frozen/trainable parameters |
| scispaCy/general biomedical extraction concern | Current pipeline uses deterministic PrimeKG lexical linker and validation bundle; official RadGraph/CheXbert outputs can complement it | Describe linker transparently and report validation; avoid relying on linker-only metric as sole evidence |
| Title typo and non-standard token labels | Docs now use Graph-Verified wording; recommended labels are ACCEPTED, FLAGGED, REJECTED | Update manuscript figures and captions |
| SOTA claim too broad | README/method docs explicitly narrow claim boundary | Remove unsupported SOTA language unless direct comparisons are run |

## Ablation Studies Required For AAAI

Run the suite produced by:

```bash
python scripts/run_ablation_suite.py \
  --manifest <MANIFEST> \
  --primekg-dir <RADIOLOGY_PRIMEKG_CACHE> \
  --output-dir <OUTPUT_DIR>/ablation_suite \
  --dataset-name <iuxray_or_mimic> \
  --generator-checkpoint-dir <VISION_T5_CHECKPOINT> \
  --split test
```

For a quick sanity check:

```bash
python scripts/run_ablation_suite.py ... --limit 100 --dry-run
```

The suite creates:

| Variant | What it isolates |
| --- | --- |
| `retrieval` | Reference-blind metadata baseline; visual retrieval is used in primary RAG runs |
| `vision_t5_standard` | Generator quality without retrieval or graph verification |
| `rag_primekg_gate` | RAG candidates plus PrimeKG/LTN/gate selection |
| `graph_constrained_balanced` | Soft graph-constrained decoding plus balanced graph/evidence selection |
| `graph_constrained_bleu_guarded` | Evidence-heavy graph-constrained profile for stronger BLEU while retaining graph audit |

## BLEU-1 > 0.5 Strategy

The defensible route to BLEU-1 above 0.5 is **not** to tune on test references.
It is to use inference-available retrieval evidence as a lexical prior, then
retain PrimeKG/LTN verification:

```text
score(c) = 0.30 * graph_score(c)
         + 0.60 * retrieval_evidence(c)
         + 0.10 * gate_acceptance(c)
```

This is implemented as the `graph_constrained_bleu_guarded` ablation. It should
be reported alongside the balanced graph profile to show the trade-off between
lexical overlap and graph consistency.

If BLEU-1 exceeds 0.5 only in the evidence-heavy setting, phrase the result as:

> Evidence-weighted graph-constrained Nesy-Gen improves lexical overlap while
> preserving PrimeKG/LTN verification.

Do not phrase it as:

> The graph alone causes BLEU improvement.

## Remaining Manual Work Before Submission

1. Run full IU-Xray ablation suite.
2. Run full MIMIC-CXR ablation suite.
3. Run official CheXbert/CheXpert and RadGraph tools on exported prediction and
   reference files.
4. Complete manual audit of the entity-linking validation sample.
5. Add qualitative examples showing candidate reports, linked entities, graph
   scores, gate decisions, and rejected unsupported terms.
6. Rewrite manuscript claims around explainability and unsupported-content
   reduction, not broad hallucination elimination.
