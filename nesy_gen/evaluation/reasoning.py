from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from nesy_gen.data.schema import RadiologyExample
from nesy_gen.evaluation.profiling import measure_latency
from nesy_gen.models.nesy_gen import NesyGenPipeline


def run_reasoning_batch(
    pipeline: NesyGenPipeline,
    examples: Iterable[RadiologyExample],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for example in examples:
        links, audit = pipeline.reason(example.indication, example.report)
        rows.append(
            {
                "study_id": example.study_id,
                "image_path": example.image_path,
                "split": example.split,
                "num_links": len(links),
                "clause_scores": audit.scores.as_dict(),
                "linked_entities": [
                    {
                        "node_name": link.node_name,
                        "node_id": link.node_id,
                        "node_type": link.node_type,
                        "negated": link.mention.negated,
                        "confidence": link.confidence,
                    }
                    for link in links
                ],
                "metadata": example.metadata,
            }
        )
    return rows


def reasoning_score_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "study_id": row["study_id"],
                "split": row["split"],
                "num_links": row["num_links"],
                **row["clause_scores"],
            }
            for row in rows
        ]
    )


def reasoning_coverage_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    coverage_rows = []
    for row in rows:
        entities = row["linked_entities"]
        positive = [entity for entity in entities if not entity["negated"]]
        negated = [entity for entity in entities if entity["negated"]]
        node_types = sorted({str(entity["node_type"]) for entity in entities})
        coverage_rows.append(
            {
                "study_id": row["study_id"],
                "split": row["split"],
                "num_links": len(entities),
                "num_positive": len(positive),
                "num_negated": len(negated),
                "node_types": ";".join(node_types),
                **row["clause_scores"],
            }
        )
    return pd.DataFrame(coverage_rows)


def save_reasoning_artifacts(
    rows: list[dict[str, object]],
    output_dir: str | Path,
    *,
    prefix: str,
    latency: dict[str, float] | None = None,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reasoning_path = out / f"{prefix}_reasoning.json"
    scores_path = out / f"{prefix}_scores.csv"
    coverage_path = out / f"{prefix}_coverage.csv"
    summary_path = out / f"{prefix}_summary.json"

    reasoning_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    scores = reasoning_score_frame(rows)
    coverage = reasoning_coverage_frame(rows)
    scores.to_csv(scores_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    summary = {
        "num_examples": len(rows),
        "reasoning_path": str(reasoning_path),
        "scores_path": str(scores_path),
        "coverage_path": str(coverage_path),
        "score_summary": scores.describe(include="all").fillna("").to_dict(),
        "coverage_summary": coverage.describe(include="all").fillna("").to_dict(),
        "latency": latency or {},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "reasoning": str(reasoning_path),
        "scores": str(scores_path),
        "coverage": str(coverage_path),
        "summary": str(summary_path),
    }


def measure_pipeline_latency(
    pipeline: NesyGenPipeline,
    example: RadiologyExample,
    *,
    warmup: int = 1,
    repeats: int = 3,
) -> dict[str, float]:
    return measure_latency(
        lambda: pipeline.reason(example.indication, example.report),
        warmup=warmup,
        repeats=repeats,
    )

