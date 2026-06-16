# Nesy-Gen

Nesy-Gen is a modular implementation scaffold for graph-verified chest X-ray
report generation. It follows the manuscript methodology:

1. Vision-Entity Cross-Attention (VECA)
2. PrimeKG patient subgraph construction with temporal/clinical edge weights
3. Logic Tensor Network-style satisfiability auditing
4. Three-layer consistency gating over graph reachability, visual grounding, and
   hallucination/NLI risk
5. Reviewer-facing evaluation for entity linking, hallucination, sensitivity, and
   latency/model size

The package is designed to clone and run in Google Colab. The public code can be
installed immediately; controlled datasets such as IU X-ray and MIMIC-CXR must
be mounted or downloaded under the terms of their licenses.

## Quick Start

```bash
git clone <your-repo-url> nesy-gen
cd nesy-gen
pip install -e ".[dev]"
python scripts/run_demo.py
python -m unittest discover -s tests
```

For PrimeKG, either download the CSV manually or run:

```bash
python scripts/download_primekg.py --out data/primekg/kg.csv
```

If you already downloaded the Harvard Dataverse bundle, place it under
`dataverse_files/` and run:

```bash
python scripts/inspect_primekg.py --dataverse-dir dataverse_files
```

The loader prefers `dataverse_files/kg.csv` and can fall back to reconstructing
the graph from `edges.csv` plus `nodes.csv`.

PrimeKG is published by the Harvard/MIMS project. Their repository describes the
ready-to-use CSV download and dataloaders: https://github.com/mims-harvard/PrimeKG

## Repository Layout

```text
nesy_gen/
  data/          Dataset schemas and JSONL loaders
  kg/            PrimeKG loading, entity linking, temporal Steiner subgraphs
  logic/         Differentiable-style fuzzy constraints and audit reports
  models/        VECA, consistency gate, and pipeline orchestration
  evaluation/    Entity, hallucination, sensitivity, and latency utilities
configs/         Default thresholds and evaluation knobs
scripts/         Colab-friendly command-line entry points
notebooks/       Reproducible Colab workflow
tests/           Smoke tests over the full neuro-symbolic path
docs/            Reviewer-driven revision plan
```

## Reviewer-Driven Additions

The initial manuscript was criticized for weak reproducibility and indirect
validation. This repo therefore includes first-class hooks for:

- entity extraction/linking accuracy against annotated mentions;
- sensitivity analysis under missed links and wrong links;
- direct hallucination accounting against reference entities;
- graph coverage reporting for missing PrimeKG concepts;
- latency and parameter-count profiling;
- GraphRAG-style retrieval baselines using the same PrimeKG interface.

## Data Expectations

Use JSONL manifests for experiments:

```json
{"study_id":"iu_0001","image_path":"/path/xray.png","indication":"shortness of breath","report":"No focal consolidation."}
```

The loaders intentionally do not redistribute IU X-ray, MIMIC-CXR, or MIMIC
metadata. Place local manifests under `data/` after obtaining dataset access.
