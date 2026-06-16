# End-to-End Experiment Pipeline

This is the paper-grade run path for reviewer-facing evidence. It replaces
manual notebook state with reproducible scripts.

## 1. Setup

```bash
git pull origin main
python -m pip install -e ".[colab,eval,dev]"
```

Mount or copy PrimeKG Dataverse files to a local folder such as:

```text
/content/drive/MyDrive/dataverse_files
```

The folder should contain `kg.csv` and ideally `nodes.csv`.

## 2. Build IU X-ray Manifest

```bash
python scripts/build_manifest.py \
  --dataset iuxray \
  --data-root /kaggle/input/chest-xrays-indiana-university \
  --output /content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1/iuxray_manifest.jsonl
```

## 3. Run PrimeKG Reasoning

For speed, build a reusable radiology-focused PrimeKG cache once:

```bash
python scripts/build_radiology_primekg.py \
  --primekg-dir /content/drive/MyDrive/dataverse_files \
  --manifest /content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1/iuxray_manifest.jsonl \
  --output-dir /content/drive/MyDrive/primekg_radiology_cache_iuxray \
  --hops 1
```

Then point reasoning at the cache directory instead of the full Dataverse folder.

Start with 50 examples:

```bash
python scripts/run_primekg_reasoning.py \
  --manifest /content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1/iuxray_manifest.jsonl \
  --primekg-dir /content/drive/MyDrive/primekg_radiology_cache_iuxray \
  --output-dir /content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1 \
  --dataset-name iuxray \
  --split test \
  --limit 50 \
  --subgraph-strategy ego \
  --latency-repeats 1
```

If this looks healthy, remove `--limit 50` for the full IU test set.

## 4. Sensitivity Analysis

```bash
python scripts/run_sensitivity_from_reasoning.py \
  --reasoning-json /content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1/iuxray_test_n50_reasoning.json \
  --output-csv /content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1/iuxray_test_n50_sensitivity.csv \
  --trials 100
```

## 5. IU X-ray Analysis, Baseline, And Factuality

Create reviewer-facing analysis tables:

```bash
python scripts/analyze_reasoning_outputs.py \
  --reasoning-json /content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1/iuxray_test_reasoning.json \
  --scores-csv /content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1/iuxray_test_scores.csv \
  --output-dir /content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1 \
  --prefix iuxray_test
```

Run a TF-IDF retrieval baseline and entity-level factuality/hallucination proxy:

```bash
python scripts/run_retrieval_baseline.py \
  --manifest /content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1/iuxray_manifest.jsonl \
  --output-csv /content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1/iuxray_retrieval_baseline_test.csv \
  --split test

python scripts/evaluate_factuality.py \
  --manifest /content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1/iuxray_manifest.jsonl \
  --predictions-csv /content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1/iuxray_retrieval_baseline_test.csv \
  --nodes-csv /content/drive/MyDrive/primekg_radiology_cache_iuxray/nodes.csv \
  --output-csv /content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1/iuxray_retrieval_factuality_test.csv
```

## 6. Reviewer-Facing Outputs

The reasoning script saves:

- `*_reasoning.json`: linked entities and clause scores per study.
- `*_scores.csv`: `num_links`, clause scores, and mean satisfaction.
- `*_coverage.csv`: positive/negated entity counts and node-type coverage.
- `*_summary.json`: aggregate tables and latency.

These outputs directly support:

- entity extraction and normalization validation;
- KG coverage and missing-concept analysis;
- logical consistency distributions;
- computational latency reporting;
- perturbation sensitivity/error propagation.

## 7. MIMIC-CXR

For the Kaggle mirror with `mimic_cxr_aug_train.csv`,
`mimic_cxr_aug_validate.csv`, and `official_data_iccv_final/files`, build a
manifest with:

```bash
python scripts/build_manifest.py \
  --dataset mimic_aug \
  --data-root /root/.cache/kagglehub/datasets/simhadrisadaram/mimic-cxr-dataset/versions/2 \
  --output /content/drive/MyDrive/mimic_dynamic_graph_outputs/flan_t5_small_run1/mimic_manifest.jsonl
```

Then build a MIMIC-specific radiology PrimeKG cache and run reasoning:

```bash
python scripts/build_radiology_primekg.py \
  --primekg-dir /content/drive/MyDrive/dataverse_files \
  --manifest /content/drive/MyDrive/mimic_dynamic_graph_outputs/flan_t5_small_run1/mimic_manifest.jsonl \
  --output-dir /content/drive/MyDrive/primekg_radiology_cache_mimic \
  --hops 1

python scripts/run_primekg_reasoning.py \
  --manifest /content/drive/MyDrive/mimic_dynamic_graph_outputs/flan_t5_small_run1/mimic_manifest.jsonl \
  --primekg-dir /content/drive/MyDrive/primekg_radiology_cache_mimic \
  --output-dir /content/drive/MyDrive/mimic_dynamic_graph_outputs/flan_t5_small_run1 \
  --dataset-name mimic \
  --split test \
  --limit 50 \
  --subgraph-strategy ego \
  --latency-repeats 1
```
