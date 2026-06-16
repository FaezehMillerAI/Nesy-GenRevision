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

## End-to-End Experiment Scripts

The paper-grade run path is documented in
`docs/end_to_end_experiment.md`. The main commands are:

```bash
python scripts/build_manifest.py --dataset iuxray --data-root <IU_ROOT> --output <OUT>/iuxray_manifest.jsonl
python scripts/build_radiology_primekg.py --primekg-dir <FULL_PRIMEKG_DIR> --manifest <OUT>/iuxray_manifest.jsonl --output-dir <PRIMEKG_RAD_CACHE> --hops 1
python scripts/run_primekg_reasoning.py --manifest <OUT>/iuxray_manifest.jsonl --primekg-dir <PRIMEKG_RAD_CACHE> --output-dir <OUT> --dataset-name iuxray --split test --limit 50 --subgraph-strategy ego --latency-repeats 1
python scripts/run_sensitivity_from_reasoning.py --reasoning-json <OUT>/iuxray_test_n50_reasoning.json --output-csv <OUT>/iuxray_test_n50_sensitivity.csv
python scripts/train_report_generator.py --manifest <OUT>/iuxray_manifest.jsonl --output-dir <OUT>/checkpoints/swin_tiny_distilgpt2_smoke --epochs 1 --max-train-examples 128 --max-val-examples 32
python scripts/generate_reports.py --manifest <OUT>/iuxray_manifest.jsonl --checkpoint-dir <OUT>/checkpoints/swin_tiny_distilgpt2_smoke --output-csv <OUT>/iuxray_generated_test_smoke.csv --split test --limit 100
python scripts/evaluate_generation.py --manifest <OUT>/iuxray_manifest.jsonl --predictions-csv <OUT>/iuxray_generated_test_smoke.csv --nodes-csv <PRIMEKG_RAD_CACHE>/nodes.csv --output-json <OUT>/iuxray_generated_test_smoke_metrics.json
python scripts/compare_generation_systems.py --manifest <OUT>/iuxray_manifest.jsonl --nodes-csv <PRIMEKG_RAD_CACHE>/nodes.csv --output-json <OUT>/generation_system_comparison.json --system neural <OUT>/iuxray_generated_test_smoke.csv --system retrieval <OUT>/iuxray_retrieval_baseline_test.csv
python scripts/train_blip_report_generator.py --manifest <OUT>/iuxray_manifest.jsonl --output-dir <OUT>/checkpoints/blip_base_iuxray_1k --model-name Salesforce/blip-image-captioning-base --epochs 3 --max-train-examples 1000 --max-val-examples 200
python scripts/generate_blip_reports.py --manifest <OUT>/iuxray_manifest.jsonl --checkpoint-dir <OUT>/checkpoints/blip_base_iuxray_1k --output-csv <OUT>/iuxray_blip_base_1k_generated_test.csv --split test
python scripts/build_qualitative_report.py --manifest <OUT>/iuxray_manifest.jsonl --predictions-csv <OUT>/iuxray_generated_test_smoke.csv --output-html <OUT>/iuxray_generated_test_smoke_qualitative.html --run-name iuxray_generated_test_smoke
python scripts/plot_results.py --output-dir <OUT>/figures --run-name iuxray_generated_test_smoke --graph-scores-csv <OUT>/iuxray_test_scores.csv --factuality-csv <OUT>/iuxray_generated_test_smoke_factuality.csv --sensitivity-csv <OUT>/iuxray_test_sensitivity.csv --entities-csv <OUT>/iuxray_test_positive_entity_frequencies.csv
```

## Data Expectations

Use JSONL manifests for experiments:

```json
{"study_id":"iu_0001","image_path":"/path/xray.png","indication":"shortness of breath","report":"No focal consolidation."}
```

The loaders intentionally do not redistribute IU X-ray, MIMIC-CXR, or MIMIC
metadata. Place local manifests under `data/` after obtaining dataset access.
