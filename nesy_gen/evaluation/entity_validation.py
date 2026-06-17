from __future__ import annotations

from collections import Counter
import json
import random
from pathlib import Path

import pandas as pd

from nesy_gen.data.schema import RadiologyExample
from nesy_gen.evaluation.clinical_metrics import CHEXPERT_CONDITIONS, chexpert_lite_labels
from nesy_gen.evaluation.generation_metrics import tokenize
from nesy_gen.kg.entity_linking import LexicalEntityLinker, LinkedEntity, normalize_text


def links_to_frame(
    examples: list[RadiologyExample],
    linker: LexicalEntityLinker,
    *,
    linker_name: str,
) -> pd.DataFrame:
    rows = []
    for example in examples:
        text = f"{example.indication} {example.report}".strip()
        for link in linker.link_text(text):
            rows.append(_link_row(example, link, text=text, linker_name=linker_name))
    return pd.DataFrame(rows)


def coverage_by_report(
    examples: list[RadiologyExample],
    filtered_links: pd.DataFrame,
    raw_links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    filtered_groups = _group_links(filtered_links)
    raw_groups = _group_links(raw_links) if raw_links is not None else {}
    rows = []
    for example in examples:
        filtered = filtered_groups.get(example.study_id, [])
        raw = raw_groups.get(example.study_id, [])
        rows.append(
            {
                "study_id": example.study_id,
                "split": example.split,
                "report_tokens": len(tokenize(example.report)),
                "filtered_links": len(filtered),
                "filtered_positive": sum(1 for row in filtered if not row["negated"]),
                "filtered_negated": sum(1 for row in filtered if row["negated"]),
                "filtered_node_types": ";".join(sorted({str(row["node_type"]) for row in filtered})),
                "raw_links": len(raw),
                "removed_by_filter": max(0, len(raw) - len(filtered)),
            }
        )
    return pd.DataFrame(rows)


def linker_ablation_frame(raw_links: pd.DataFrame, filtered_links: pd.DataFrame) -> pd.DataFrame:
    raw_keys = _link_key_counts(raw_links)
    filtered_keys = _link_key_counts(filtered_links)
    rows = []
    for key, raw_count in raw_keys.items():
        filtered_count = filtered_keys.get(key, 0)
        if raw_count == filtered_count:
            continue
        node_id, node_name, node_type, mention_text = key
        rows.append(
            {
                "node_id": node_id,
                "node_name": node_name,
                "node_type": node_type,
                "mention_text": mention_text,
                "raw_count": raw_count,
                "filtered_count": filtered_count,
                "removed_count": raw_count - filtered_count,
            }
        )
    return pd.DataFrame(rows).sort_values("removed_count", ascending=False) if rows else pd.DataFrame()


def entity_frequency_frame(links: pd.DataFrame) -> pd.DataFrame:
    if links.empty:
        return pd.DataFrame(columns=["node_id", "node_name", "node_type", "negated", "count"])
    return (
        links.groupby(["node_id", "node_name", "node_type", "negated"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def condition_link_coverage_frame(
    examples: list[RadiologyExample],
    filtered_links: pd.DataFrame,
) -> pd.DataFrame:
    groups = _group_links(filtered_links)
    rows = []
    for example in examples:
        labels = chexpert_lite_labels(example.report)
        linked_text = " ".join(
            [
                str(row["node_name"])
                for row in groups.get(example.study_id, [])
                if not bool(row["negated"])
            ]
        )
        norm_linked_text = normalize_text(linked_text)
        for condition, label in labels.items():
            if label == -1:
                continue
            aliases = CHEXPERT_CONDITIONS[condition]
            linked = any(normalize_text(alias) in norm_linked_text for alias in aliases)
            rows.append(
                {
                    "study_id": example.study_id,
                    "split": example.split,
                    "condition": condition,
                    "reference_label": label,
                    "linked_by_primekg": linked,
                }
            )
    return pd.DataFrame(rows)


def manual_audit_sample(
    links: pd.DataFrame,
    *,
    sample_size: int = 100,
    seed: int = 13,
) -> pd.DataFrame:
    if links.empty:
        return links
    rng = random.Random(seed)
    indices = list(links.index)
    rng.shuffle(indices)
    sampled = links.loc[indices[:sample_size]].copy()
    sampled["mention_correct"] = ""
    sampled["primekg_node_correct"] = ""
    sampled["negation_correct"] = ""
    sampled["notes"] = ""
    return sampled[
        [
            "study_id",
            "split",
            "mention_text",
            "node_name",
            "node_id",
            "node_type",
            "negated",
            "context",
            "mention_correct",
            "primekg_node_correct",
            "negation_correct",
            "notes",
        ]
    ]


def write_entity_validation_bundle(
    examples: list[RadiologyExample],
    vocab: pd.DataFrame,
    output_dir: str | Path,
    *,
    prefix: str,
    audit_sample_size: int = 100,
    seed: int = 13,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filtered_linker = LexicalEntityLinker(vocab)
    raw_linker = LexicalEntityLinker(vocab, blocked_aliases=[], blocked_node_type_fragments=[])

    filtered_links = links_to_frame(examples, filtered_linker, linker_name="filtered")
    raw_links = links_to_frame(examples, raw_linker, linker_name="raw")
    coverage = coverage_by_report(examples, filtered_links, raw_links)
    ablation = linker_ablation_frame(raw_links, filtered_links)
    frequencies = entity_frequency_frame(filtered_links)
    condition_coverage = condition_link_coverage_frame(examples, filtered_links)
    audit = manual_audit_sample(filtered_links, sample_size=audit_sample_size, seed=seed)

    paths = {
        "filtered_links": out / f"{prefix}_filtered_links.csv",
        "raw_links": out / f"{prefix}_raw_links.csv",
        "coverage": out / f"{prefix}_entity_coverage_by_report.csv",
        "linker_ablation": out / f"{prefix}_linker_filtering_ablation.csv",
        "entity_frequencies": out / f"{prefix}_entity_frequencies.csv",
        "condition_link_coverage": out / f"{prefix}_condition_link_coverage.csv",
        "manual_audit_sample": out / f"{prefix}_manual_entity_audit_sample.csv",
        "summary": out / f"{prefix}_entity_validation_summary.json",
    }
    filtered_links.to_csv(paths["filtered_links"], index=False)
    raw_links.to_csv(paths["raw_links"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    ablation.to_csv(paths["linker_ablation"], index=False)
    frequencies.to_csv(paths["entity_frequencies"], index=False)
    condition_coverage.to_csv(paths["condition_link_coverage"], index=False)
    audit.to_csv(paths["manual_audit_sample"], index=False)

    summary = {
        "num_examples": len(examples),
        "filtered_total_links": int(len(filtered_links)),
        "raw_total_links": int(len(raw_links)),
        "removed_by_filter": int(max(0, len(raw_links) - len(filtered_links))),
        "mean_filtered_links_per_report": float(coverage["filtered_links"].mean()) if not coverage.empty else 0.0,
        "mean_raw_links_per_report": float(coverage["raw_links"].mean()) if not coverage.empty else 0.0,
        "reports_with_no_filtered_links": int((coverage["filtered_links"] == 0).sum()) if not coverage.empty else 0,
        "condition_rows": int(len(condition_coverage)),
        "condition_link_coverage_rate": (
            float(condition_coverage["linked_by_primekg"].mean()) if not condition_coverage.empty else 0.0
        ),
        **{key: str(value) for key, value in paths.items() if key != "summary"},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}


def _link_row(
    example: RadiologyExample,
    link: LinkedEntity,
    *,
    text: str,
    linker_name: str,
) -> dict[str, object]:
    start = max(0, link.mention.start - 60)
    end = min(len(text), link.mention.end + 60)
    return {
        "study_id": example.study_id,
        "split": example.split,
        "linker": linker_name,
        "mention_text": link.mention.text,
        "mention_start": link.mention.start,
        "mention_end": link.mention.end,
        "node_id": link.node_id,
        "node_name": link.node_name,
        "node_type": link.node_type,
        "confidence": link.confidence,
        "negated": link.mention.negated,
        "context": text[start:end],
    }


def _group_links(links: pd.DataFrame | None) -> dict[str, list[dict[str, object]]]:
    if links is None or links.empty:
        return {}
    groups: dict[str, list[dict[str, object]]] = {}
    for row in links.to_dict(orient="records"):
        groups.setdefault(str(row["study_id"]), []).append(row)
    return groups


def _link_key_counts(links: pd.DataFrame) -> Counter[tuple[str, str, str, str]]:
    counter: Counter[tuple[str, str, str, str]] = Counter()
    if links.empty:
        return counter
    for row in links.itertuples(index=False):
        counter[
            (
                str(row.node_id),
                str(row.node_name),
                str(row.node_type),
                str(row.mention_text),
            )
        ] += 1
    return counter
