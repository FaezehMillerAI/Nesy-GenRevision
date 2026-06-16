from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from nesy_gen.data.schema import RadiologyExample, write_jsonl


def clean_report_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("XXXX", "").split())


def split_ids(ids: list[object], *, seed: int = 13) -> dict[object, str]:
    """Return deterministic 7:1:2 train/val/test splits."""

    rng = random.Random(seed)
    unique_ids = list(dict.fromkeys(ids))
    rng.shuffle(unique_ids)
    n = len(unique_ids)
    train_end = int(0.7 * n)
    val_end = train_end + int(0.1 * n)
    split_map: dict[object, str] = {}
    for idx, value in enumerate(unique_ids):
        if idx < train_end:
            split_map[value] = "train"
        elif idx < val_end:
            split_map[value] = "val"
        else:
            split_map[value] = "test"
    return split_map


def build_iuxray_manifest(
    data_root: str | Path,
    output_path: str | Path,
    *,
    seed: int = 13,
    projection: str = "frontal",
) -> list[RadiologyExample]:
    root = Path(data_root)
    reports_path = root / "indiana_reports.csv"
    projections_path = root / "indiana_projections.csv"
    if not reports_path.exists() or not projections_path.exists():
        raise FileNotFoundError(
            f"Expected indiana_reports.csv and indiana_projections.csv under {root}"
        )

    reports = pd.read_csv(reports_path)
    projections = pd.read_csv(projections_path)
    selected = projections[projections["projection"].str.lower().eq(projection.lower())].copy()

    image_files = {path.name: path for path in root.rglob("*.png")}
    merged = reports.merge(selected[["uid", "filename", "projection"]], on="uid", how="inner")
    split_map = split_ids(merged["uid"].drop_duplicates().tolist(), seed=seed)

    examples: list[RadiologyExample] = []
    for row in merged.itertuples(index=False):
        findings = clean_report_text(getattr(row, "findings", ""))
        impression = clean_report_text(getattr(row, "impression", ""))
        report = f"{findings} {impression}".strip()
        image_path = image_files.get(str(row.filename))
        if not report or image_path is None:
            continue
        uid = getattr(row, "uid")
        examples.append(
            RadiologyExample(
                study_id=f"iu_{uid}",
                image_path=str(image_path),
                indication=clean_report_text(getattr(row, "indication", "")),
                report=report,
                split=split_map[uid],
                metadata={
                    "uid": int(uid),
                    "filename": str(row.filename),
                    "projection": str(row.projection),
                    "mesh": clean_report_text(getattr(row, "MeSH", "")),
                    "problems": clean_report_text(getattr(row, "Problems", "")),
                },
            )
        )

    write_jsonl(output_path, examples)
    return examples


def build_generic_csv_manifest(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    data_root: str | Path | None = None,
    study_id_col: str = "study_id",
    image_path_col: str = "image_path",
    report_col: str = "report",
    indication_col: str = "indication",
    split_col: str | None = None,
    seed: int = 13,
) -> list[RadiologyExample]:
    """Build a manifest from a normalized CSV or a discovered MIMIC-style table."""

    frame = pd.read_csv(csv_path)
    required = {study_id_col, report_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CSV manifest source is missing required columns: {missing}")

    if split_col and split_col in frame.columns:
        split_map = {row[study_id_col]: row[split_col] for _, row in frame.iterrows()}
    else:
        split_map = split_ids(frame[study_id_col].drop_duplicates().tolist(), seed=seed)

    root = Path(data_root) if data_root else None
    examples: list[RadiologyExample] = []
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        study_id = str(row_dict[study_id_col])
        image_value = clean_report_text(row_dict.get(image_path_col, ""))
        if image_value and root and not Path(image_value).is_absolute():
            image_value = str(root / image_value)
        examples.append(
            RadiologyExample(
                study_id=study_id,
                image_path=image_value or None,
                indication=clean_report_text(row_dict.get(indication_col, "")),
                report=clean_report_text(row_dict[report_col]),
                split=str(split_map[study_id]),
                metadata={},
            )
        )

    write_jsonl(output_path, examples)
    return examples

