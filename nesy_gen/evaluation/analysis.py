from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pandas as pd


def load_reasoning_rows(path: str | Path) -> list[dict[str, object]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def entity_frequency_frame(rows: list[dict[str, object]], *, include_negated: bool = True) -> pd.DataFrame:
    counter: Counter[tuple[str, str, str, bool]] = Counter()
    for row in rows:
        for entity in row.get("linked_entities", []):
            negated = bool(entity.get("negated", False))
            if negated and not include_negated:
                continue
            key = (
                str(entity.get("node_id", "")),
                str(entity.get("node_name", "")),
                str(entity.get("node_type", "")),
                negated,
            )
            counter[key] += 1
    return pd.DataFrame(
        [
            {
                "node_id": node_id,
                "node_name": node_name,
                "node_type": node_type,
                "negated": negated,
                "count": count,
            }
            for (node_id, node_name, node_type, negated), count in counter.most_common()
        ]
    )


def low_score_frame(rows: list[dict[str, object]], *, n: int = 50) -> pd.DataFrame:
    records = []
    for row in rows:
        scores = row.get("clause_scores", {})
        entities = row.get("linked_entities", [])
        records.append(
            {
                "study_id": row.get("study_id"),
                "num_links": row.get("num_links", len(entities)),
                "num_positive": sum(1 for entity in entities if not entity.get("negated", False)),
                "num_negated": sum(1 for entity in entities if entity.get("negated", False)),
                "bio_temporal": scores.get("bio_temporal", 0.0),
                "finding_to_diagnosis": scores.get("finding_to_diagnosis", 0.0),
                "located_in_type": scores.get("located_in_type", 0.0),
                "mean": scores.get("mean", 0.0),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.sort_values(["mean", "num_links"], ascending=[True, True]).head(n)


def score_bin_frame(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    bins = [-0.001, 0.25, 0.5, 0.65, 0.8, 1.0]
    labels = ["0-0.25", "0.25-0.50", "0.50-0.65", "0.65-0.80", "0.80-1.00"]
    frame = scores.copy()
    frame["mean_bin"] = pd.cut(frame["mean"], bins=bins, labels=labels)
    return (
        frame.groupby("mean_bin", observed=False)
        .agg(
            examples=("study_id", "count"),
            mean_links=("num_links", "mean"),
            mean_bio_temporal=("bio_temporal", "mean"),
            mean_finding_to_diagnosis=("finding_to_diagnosis", "mean"),
            mean_located_in_type=("located_in_type", "mean"),
        )
        .reset_index()
    )


def write_analysis_bundle(
    reasoning_json: str | Path,
    scores_csv: str | Path,
    output_dir: str | Path,
    *,
    prefix: str,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_reasoning_rows(reasoning_json)
    scores = pd.read_csv(scores_csv)

    entities_path = out / f"{prefix}_entity_frequencies.csv"
    positive_entities_path = out / f"{prefix}_positive_entity_frequencies.csv"
    low_scores_path = out / f"{prefix}_lowest_score_examples.csv"
    bins_path = out / f"{prefix}_score_bins.csv"
    bundle_path = out / f"{prefix}_analysis_summary.json"

    entity_frequency_frame(rows).to_csv(entities_path, index=False)
    entity_frequency_frame(rows, include_negated=False).to_csv(positive_entities_path, index=False)
    low_score_frame(rows).to_csv(low_scores_path, index=False)
    score_bin_frame(scores).to_csv(bins_path, index=False)

    summary = {
        "num_reasoning_rows": len(rows),
        "mean_score": float(scores["mean"].mean()) if "mean" in scores else 0.0,
        "median_score": float(scores["mean"].median()) if "mean" in scores else 0.0,
        "zero_score_examples": int((scores["mean"] == 0).sum()) if "mean" in scores else 0,
        "entity_frequencies": str(entities_path),
        "positive_entity_frequencies": str(positive_entities_path),
        "lowest_score_examples": str(low_scores_path),
        "score_bins": str(bins_path),
    }
    bundle_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "summary": str(bundle_path),
        "entity_frequencies": str(entities_path),
        "positive_entity_frequencies": str(positive_entities_path),
        "lowest_score_examples": str(low_scores_path),
        "score_bins": str(bins_path),
    }

