# Nesy-Gen

Nesy-Gen is a neuro-symbolic framework for chest X-ray report generation with
PrimeKG-grounded reasoning. The repository is organized around one paper-grade
pipeline:

1. train a frozen-vision image-to-report generator;
2. retrieve clinically similar report evidence;
3. link report entities to PrimeKG;
4. verify candidates with Logic Tensor Network-style graph constraints;
5. select the final report with a consistency gate;
6. evaluate generation quality, graph consistency, entity linking, and official
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

For training-free MedGemma drafting with frozen MedSigLIP visual retrieval:

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
