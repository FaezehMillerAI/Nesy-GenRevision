from __future__ import annotations

import ast
import json
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
    selected = selected.sort_values(["uid", "filename"]).drop_duplicates("uid", keep="first")

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


def build_r2gen_iuxray_manifest(
    data_root: str | Path,
    output_path: str | Path,
    *,
    one_example_per_image: bool = True,
) -> list[RadiologyExample]:
    """Build a manifest from the R2Gen IU-Xray `annotation.json` layout.

    Common layouts supported:
    - `<data_root>/annotation.json` and `<data_root>/images/...`
    - `<data_root>/iu_xray/annotation.json` and `<data_root>/iu_xray/images/...`

    The R2Gen annotation stores predefined `train`, `val`, and `test` splits
    and each study may contain multiple image paths. By default we follow the
    original notebook convention and create one example per image.
    """

    root = Path(data_root)
    annotation_path = _find_r2gen_annotation(root)
    annotation_root = annotation_path.parent
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not all(split in data for split in ("train", "val", "test")):
        raise ValueError(f"Expected train/val/test keys in {annotation_path}")

    examples: list[RadiologyExample] = []
    for split in ("train", "val", "test"):
        for row in data[split]:
            report = clean_report_text(row.get("report", ""))
            if not report:
                continue
            study_id = str(row.get("id") or row.get("uid") or f"{split}_{len(examples)}")
            image_paths = row.get("image_path") or row.get("images") or []
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            if not image_paths:
                continue
            selected_paths = image_paths if one_example_per_image else image_paths[:1]
            for image_idx, image_value in enumerate(selected_paths):
                image_path = _resolve_r2gen_image_path(annotation_root, str(image_value))
                if image_path is None:
                    continue
                suffix = f"_{image_idx}" if one_example_per_image and len(selected_paths) > 1 else ""
                examples.append(
                    RadiologyExample(
                        study_id=f"r2gen_{study_id}{suffix}",
                        image_path=str(image_path),
                        indication=clean_report_text(row.get("indication", "")),
                        report=report,
                        split=split,
                        metadata={
                            "source": "r2gen_iuxray",
                            "r2gen_id": study_id,
                            "image_path": str(image_value),
                            "image_index": image_idx,
                        },
                    )
                )

    write_jsonl(output_path, examples)
    return examples


def build_mimic_aug_manifest(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    seed: int = 13,
    validate_test_fraction: float = 0.5,
) -> list[RadiologyExample]:
    """Build a manifest from the Kaggle MIMIC-CXR augmented CSV mirror.

    Expected files:
    - `mimic_cxr_aug_train.csv`
    - `mimic_cxr_aug_validate.csv`

    The mirror stores image paths and report text as Python-list-like strings.
    We create one row per available report text, pair it with a preferred
    frontal image from the same subject row, and split the validation CSV into
    validation/test partitions deterministically.
    """

    root = Path(dataset_root)
    if not (root / "mimic_cxr_aug_train.csv").exists() and root.name == "official_data_iccv_final":
        root = root.parent
    train_path = root / "mimic_cxr_aug_train.csv"
    validate_path = root / "mimic_cxr_aug_validate.csv"
    if not train_path.exists() or not validate_path.exists():
        raise FileNotFoundError(
            f"Expected mimic_cxr_aug_train.csv and mimic_cxr_aug_validate.csv under {root}"
        )

    train = pd.read_csv(train_path)
    validate = pd.read_csv(validate_path)
    validation_subjects = validate["subject_id"].drop_duplicates().tolist()
    rng = random.Random(seed)
    rng.shuffle(validation_subjects)
    test_subjects = set(validation_subjects[: int(len(validation_subjects) * validate_test_fraction)])

    examples: list[RadiologyExample] = []
    examples.extend(_mimic_rows_to_examples(train, root, split="train"))
    for example in _mimic_rows_to_examples(validate, root, split="val"):
        subject_id = example.metadata["subject_id"]
        split = "test" if subject_id in test_subjects else "val"
        examples.append(
            RadiologyExample(
                study_id=example.study_id,
                image_path=example.image_path,
                indication=example.indication,
                report=example.report,
                split=split,
                metadata=example.metadata,
            )
        )

    write_jsonl(output_path, examples)
    return examples


def _find_r2gen_annotation(root: Path) -> Path:
    candidates = [
        root / "annotation.json",
        root / "iu_xray" / "annotation.json",
        root / "iu_xray" / "iu_xray" / "annotation.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(root.rglob("annotation.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find R2Gen annotation.json under {root}")


def _resolve_r2gen_image_path(annotation_root: Path, image_value: str) -> Path | None:
    value = Path(image_value)
    candidates = []
    if value.is_absolute():
        candidates.append(value)
    else:
        candidates.extend(
            [
                annotation_root / value,
                annotation_root / "images" / value,
                annotation_root / "iu_xray" / "images" / value,
                annotation_root.parent / value,
                annotation_root.parent / "images" / value,
                annotation_root.parent / "iu_xray" / "images" / value,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _mimic_rows_to_examples(frame: pd.DataFrame, dataset_root: Path, *, split: str) -> list[RadiologyExample]:
    examples: list[RadiologyExample] = []
    files_root = dataset_root / "official_data_iccv_final"
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        subject_id = int(row_dict["subject_id"])
        report_texts = [clean_report_text(text) for text in parse_list_cell(row_dict.get("text", []))]
        report_texts = [text for text in report_texts if text]
        if not report_texts:
            continue

        image_candidates = _preferred_mimic_images(row_dict)
        if not image_candidates:
            continue
        image_path = files_root / image_candidates[0]
        if not image_path.exists():
            image_path = dataset_root / image_candidates[0]

        study_id = _study_id_from_path(image_candidates[0]) or f"subject_{subject_id}"
        for report_idx, report in enumerate(report_texts):
            suffix = "" if len(report_texts) == 1 else f"_{report_idx}"
            examples.append(
                RadiologyExample(
                    study_id=f"mimic_{study_id}{suffix}",
                    image_path=str(image_path),
                    indication="",
                    report=report,
                    split=split,
                    metadata={
                        "subject_id": subject_id,
                        "study_id": study_id,
                        "report_index": report_idx,
                        "source": "mimic_cxr_aug",
                    },
                )
            )
    return examples


def parse_list_cell(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).lower() != "nan"]
    if pd.isna(value):
        return []
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text] if text and text.lower() != "nan" else []
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).lower() != "nan"]
    return [str(parsed)] if str(parsed).lower() != "nan" else []


def _preferred_mimic_images(row_dict: dict[str, object]) -> list[str]:
    for column in ("PA", "AP", "image", "Lateral"):
        values = parse_list_cell(row_dict.get(column, []))
        values = [value for value in values if value.lower().endswith((".jpg", ".jpeg", ".png"))]
        if values:
            return values
    return []


def _study_id_from_path(path: str) -> str | None:
    for part in Path(path).parts:
        if part.startswith("s") and part[1:].isdigit():
            return part
    return None
