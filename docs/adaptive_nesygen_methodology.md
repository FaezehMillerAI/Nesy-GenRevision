# Adaptive NeSy-Gen Methodology

## Scope

Adaptive NeSy-Gen reuses the frozen-vision report generator and leakage-safe
visual RAG index. It changes inference, not the previously trained checkpoint.
The method treats explanation as an inference artifact rather than asking an
LLM to rationalize a completed report.

The drafting component is switchable. The checkpoint-based Vision-T5 baseline
can be replaced by MedGemma or the smaller chest-X-ray-specific CheXagent 3B,
with frozen MedSigLIP retrieval. Both multimodal backends support task-specific
decoder QLoRA while their medical vision encoders remain frozen. Retrieved
reports are training-split evidence only, and the same adaptive verifier is
applied. Because the official model cards list MIMIC-CXR among the relevant
pretraining or adaptation sources, MIMIC experiments are not described as
strict zero-shot experiments.

## Inference

1. The image-conditioned generator produces report candidates.
2. Visual RAG retrieves training reports, excluding the same underlying study.
3. The highest-ranked generated candidate becomes the draft. Retrieval reports
   remain evidence and never expose the test reference.
4. The draft is segmented into clinical claims and linked to PrimeKG entities
   with assertion polarity.
5. A claim supported by at least `m` retrieved studies with support above
   `tau_fast` follows the fast path.
6. Other linked claims invoke a patient/report-specific PrimeKG subgraph, LTN
   clause scoring, and the Consistency Gate.
7. A disputed claim may be replaced only by a retrieved training sentence that
   has the same set of linked entity identifiers and negation polarity and
   exceeds `tau_revise`. Otherwise it is preserved and flagged for review.

This conservative policy avoids introducing a new clinical entity during
revision. It also keeps report edits measurable and auditable.

## Evidence Contract

Each claim trace contains:

- original and final claim text;
- linked PrimeKG identifiers, types, assertion polarity, and linker confidence;
- visual and retrieval support;
- whether graph verification was triggered;
- LTN clause truth values and aggregate truth;
- PrimeKG node status and an explanation path;
- gate confidence, decision, reason, replacement provenance, and latency.

The trace is faithful in the procedural sense: its values are consumed by the
router or gate during inference. This is narrower than claiming that the trace
is a complete causal explanation of the neural generator.

## Primary Claims

The defensible primary claims are adaptive neuro-symbolic verification,
claim-level provenance, process transparency, and reduced verification cost
relative to always-on claim verification. Hallucination reduction remains a
secondary empirical question and should be claimed only if official clinical
metrics and expert review support it.

## Required Ablations

| Variant | Purpose |
| --- | --- |
| Raw generator | Neural baseline |
| Visual RAG | Retrieval contribution |
| Report-level PrimeKG/LTN gate | Prior Nesy-Gen inference |
| Adaptive audit-only | Routing and explanation without editing |
| Adaptive revision | Proposed full method |
| Always-on claim verification | Efficiency control |
| Shuffled or relation-ablated graph | PrimeKG structure control |
| No LTN | Logic contribution |
| No gate | Gate contribution |

Report generation metrics, official CheXbert/CheXpert and RadGraph scores,
entity-linking validation, leakage audits, escalation rate, graph calls, latency,
and qualitative claim traces must be reported together.
