# Nesy-Gen

Adaptive NeSy-Gen is a neuro-symbolic framework for chest X-ray report generation
with claim-level PrimeKG-grounded reasoning. The repository is organized around
one paper-grade pipeline:

1. select either a trained frozen-vision generator or the official MedGemma
   Hugging Face checkpoint without task-specific fine-tuning;
2. retrieve visually similar training studies with a frozen image encoder;
3. decompose the draft into claims and link clinical entities to PrimeKG;
4. route high-consensus claims through a fast path and uncertain linked claims
   through LTN-style graph constraints;
5. preserve accepted claims and selectively revise only from evidence satisfying
   the exact entity-and-polarity contract;
6. export inference-faithful claim traces and evaluate generation quality,
   efficiency, graph consistency, entity linking, and official
   clinical metrics.

The intended AAAI framing is **explainable, graph-grounded report generation**.
Hallucination is measured as an entity-level and clinical-label proxy, but the
method should not be claimed to eliminate hallucinations without official metric
outputs and expert review.

## Main Novelties

- **Ante-hoc PrimeKG reasoning:** candidate reports are checked against a
  patient/report-specific PrimeKG subgraph before final selection.
- **Soft graph-constrained decoding:** supported PrimeKG entity terms can be
  encouraged during generation without hard-coding report templates.
- **LTN-style verification:** biological/temporal connectivity,
  finding-diagnosis support, and location/type compatibility are scored as fuzzy
  clauses.
- **Consistency gate:** graph satisfaction, retrieval evidence, entity
  grounding, and risk flags are combined before a report is emitted.
- **Reviewer-facing validation:** the repo exports entity-linking audit tables,
  sensitivity analysis, leakage checks, qualitative examples, and official metric
  inputs.

## Recommended Colab Entry Point

Use the simplified notebook for normal experiments:

```text
notebooks/AAAI_NesyGen_Simple_Colab.ipynb
```

The notebook exposes only the main switches:

- `RUN_DATASET`: `iuxray_official` or `mimic_aug`
- `RUN_SIZE`: `smoke` or `full`
- `USE_GRAPH_CONSTRAINTS`
- `USE_GRAPH_AWARE_TRAINING`

The advanced notebook remains available for ablations:

```text
notebooks/AAAI_RAG_PrimeKG_LTN_Gate_Colab.ipynb
```

To reuse a trained checkpoint with the proposed adaptive, claim-level method:

```text
notebooks/AAAI_Adaptive_NesyGen_Colab.ipynb
```

For MedGemma drafting without task-specific fine-tuning and frozen MedSigLIP
visual retrieval:

```text
notebooks/AAAI_Adaptive_MedGemma_Colab.ipynb
```

This notebook supports `zero_shot` and retrieval-conditioned `few_shot` modes.
Both official Google checkpoints are gated on Hugging Face. Their terms must be
accepted before use. Since their documented pretraining data includes
MIMIC-CXR, MIMIC results are labelled **no task-specific fine-tuning**, not
strict unseen-data zero-shot.

This mode fast-accepts high-consensus claims, invokes PrimeKG/LTN only for
uncertain claims, and records the evidence and gate decision actually used at
inference. Selective revision is extractive and may only use a visually
retrieved training sentence with the same linked entities and assertion
polarity. The original report-level pipeline remains available as a baseline.

The default Hugging Face models are `google/medgemma-4b-it` for image-conditioned
Findings drafting and `google/medsiglip-448` for frozen visual retrieval. Accept
the Health AI Developer Foundations terms for both model repositories before
running them. These are research models and require task- and site-specific
validation; this repository does not present their output as clinical advice.

Example no-task-specific-fine-tuning run:

```bash
python scripts/generate_medgemma_adaptive_reports.py \
  --manifest <MANIFEST.jsonl> \
  --primekg-dir <TRAIN_SEEDED_RADIOLOGY_PRIMEKG_CACHE> \
  --dataset-name iuxray \
  --draft-mode few_shot \
  --retrieval-cache <CACHE_DIR>/medsiglip_train_index.npz \
  --output-csv <OUTPUT_DIR>/predictions.csv \
  --candidates-csv <OUTPUT_DIR>/retrieved_candidates.csv \
  --claim-trace-jsonl <OUTPUT_DIR>/claim_traces.jsonl \
  --claim-audit-csv <OUTPUT_DIR>/claim_audit.csv
```

The command also writes a `.run.json` companion containing model identifiers,
thresholds, index build/load time, retrieval timing, and index size. Prediction
rows contain generation, verification, end-to-end latency, graph calls,
all-claim and linked-claim escalation rates, and peak allocated GPU memory.

## Compact radiology QLoRA fine-tuning

Use the dedicated Colab workflow for task-specific Findings adaptation:

```text
notebooks/AAAI_MedGemma_QLoRA_Finetuning_Colab.ipynb
```

The notebook defaults to Stanford AIMI's 3B CheXagent Findings checkpoint, a
chest-X-ray specialist with 25% fewer parameters than MedGemma 4B. Set
`MODEL_FAMILY='medgemma'` to retain MedGemma as a controlled ablation. Both
paths use 4-bit NF4 base weights, decoder LoRA adapters, bfloat16 computation,
gradient checkpointing, and a frozen medical vision encoder. Half of the primary training examples are
retrieval-conditioned by default; their evidence comes only from visually
retrieved training studies with same-study and alternate-view exclusion.
Training and model selection consume `train` and `val` only. The notebook keeps
the final test run behind an explicit switch.

Equivalent command-line training:

```bash
python -m pip install -e ".[finetune,eval]"
python scripts/train_medgemma_lora.py \
  --manifest <MANIFEST.jsonl> \
  --output-dir <OUTPUT_DIR>/training \
  --model-family chexagent \
  --model-name StanfordAIMI/CheXagent-2-3b-srrg-findings \
  --train-split train \
  --eval-split val \
  --retrieval-cache <CACHE_DIR>/medsiglip_train_index.npz \
  --retrieval-probability 0.5 \
  --lora-rank 16 \
  --lora-alpha 16 \
  --learning-rate 2e-4 \
  --epochs 3
```

Pass the resulting adapter to inference with:

```text
--draft-backend chexagent \
--draft-model StanfordAIMI/CheXagent-2-3b-srrg-findings \
--draft-adapter <OUTPUT_DIR>/training/final_adapter
```

## Quick Start

```bash
git clone https://github.com/FaezehMillerAI/Nesy-GenRevision.git
cd Nesy-GenRevision
python -m pip install -e ".[torch,eval,colab,dev]"
python -m unittest discover -s tests
```

PrimeKG Dataverse files should be mounted locally, for example:

```text
/content/drive/MyDrive/dataverse_files
```

IU-Xray in the R2Gen-style annotation layout should look like:

```text
/content/drive/MyDrive/iuxray/
  annotation.json
  images/
    CXR.../
      0.png
      1.png
```

MIMIC-CXR augmented experiments expect the Kaggle mirror with
`mimic_cxr_aug_train.csv`, `mimic_cxr_aug_validate.csv`, and
`official_data_iccv_final/files`.

## Repository Layout

```text
nesy_gen/
  agents/         adaptive claim routing and evidence-bound revision
  baselines/      retrieval baselines and leakage-safe query construction
  data/           dataset schemas, manifest builders, Kaggle/Drive resolvers
  generation/     RAG selection and PrimeKG-constrained decoding
  kg/             PrimeKG loading, entity linking, temporal subgraphs
  logic/          fuzzy/LTN-style constraints
  models/         Vision-T5 backbone, Nesy-Gen pipeline, consistency gate
  training/       leakage-safe MedGemma multimodal QLoRA data and collation
  evaluation/     lexical, entity, clinical, graph, official-metric adapters
scripts/          command-line experiment entry points
notebooks/        Colab workflows
docs/             methodology and revision notes
tests/            unit and smoke tests
```

## Paper-Grade Evidence Checklist

For each dataset and method variant, save:

- predictions CSV and candidate audit CSV;
- quick lexical metrics plus diversity diagnostics;
- entity factuality proxy and graph satisfaction scores;
- entity-linking validation bundle;
- leakage audit;
- official CheXbert/CheXpert and RadGraph inputs/outputs;
- qualitative HTML examples with graph/gate decisions;
- faithful claim traces and adaptive-routing efficiency;
- ablations: standard generation, RAG only, RAG + PrimeKG gate, graph
  constrained decoding, adaptive audit-only, adaptive revision, and always-on
  claim verification.

See [docs/aaai_methodology.md](docs/aaai_methodology.md) for the recommended
methodology and claim boundaries. See
[docs/reviewer_response_matrix.md](docs/reviewer_response_matrix.md) for the
reviewer concern-to-artifact mapping.

The main ablation suite is:

```bash
python scripts/run_ablation_suite.py \
  --manifest <MANIFEST> \
  --primekg-dir <RADIOLOGY_PRIMEKG_CACHE> \
  --output-dir <OUTPUT_DIR>/ablation_suite \
  --dataset-name <DATASET_NAME> \
  --generator-checkpoint-dir <VISION_T5_CHECKPOINT> \
  --split test
```

For the BLEU-oriented graph setting, use the emitted
`graph_constrained_bleu_guarded` variant and report it together with the balanced
graph-constrained variant.

Primary RAG runs use frozen-image retrieval and exclude the same underlying
study, including alternate views. The query reference is never read during
retrieval or candidate filtering. Cross-split duplicate reports are reported by
the separate leakage audit instead of being removed using hidden test labels.

## Important Claim Boundary

The quick metrics named `*_lite` are development diagnostics. They are useful for
debugging and ablation triage, but final paper tables should prioritize official
COCO-caption metrics, CheXbert/CheXpert labels, RadGraph, entity-linking manual
audit, and qualitative reviewer-facing examples.
