from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    run_dataset: str
    dataset_root: Path
    data_root: Path
    output_dir: Path
    source: str = "kagglehub"
    cache_root: Path | None = None


def resolve_kaggle_dataset(
    run_dataset: str,
    *,
    cache_root: str | Path | None = None,
    populate_cache: bool = False,
) -> DatasetPaths:
    """Resolve the Colab/KaggleHub dataset layout used for the revision runs.

    If ``cache_root`` is provided, the resolver first checks for a previously
    copied dataset under that persistent directory. In Colab this should point
    to Google Drive, for example ``/content/drive/MyDrive/nesy_gen_dataset_cache``.
    """

    cache_base = Path(cache_root) if cache_root else None
    cached = _resolve_cached_dataset(run_dataset, cache_base) if cache_base else None
    if cached:
        return cached

    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - depends on Colab env
        raise ImportError("Install kagglehub or `pip install -e .[colab]`.") from exc

    if run_dataset == "mimic_cxr":
        dataset_root = Path(kagglehub.dataset_download("simhadrisadaram/mimic-cxr-dataset"))
        data_root = dataset_root / "official_data_iccv_final"
        output_dir = Path("/content/drive/MyDrive/mimic_dynamic_graph_outputs/flan_t5_small_run1")
    elif run_dataset == "iuxray":
        dataset_root = Path(kagglehub.dataset_download("raddar/chest-xrays-indiana-university"))
        data_root = dataset_root
        output_dir = Path("/content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1")
    else:
        raise ValueError("run_dataset must be one of: mimic_cxr, iuxray")

    if cache_base and populate_cache:
        dataset_root = populate_dataset_cache(
            run_dataset,
            dataset_root,
            cache_base,
        )
        data_root = _data_root_for(run_dataset, dataset_root)
        source = "drive_cache_populated"
    else:
        source = "kagglehub"

    return DatasetPaths(
        run_dataset=run_dataset,
        dataset_root=dataset_root,
        data_root=data_root,
        output_dir=output_dir,
        source=source,
        cache_root=cache_base,
    )


def print_dataset_paths(paths: DatasetPaths) -> None:
    print("Dataset:", paths.run_dataset)
    print("Source:", paths.source)
    print("Dataset root:", paths.dataset_root)
    print("Data root:", paths.data_root)
    print("Exists:", paths.data_root.exists())
    print("Output dir:", paths.output_dir)
    if paths.cache_root:
        print("Cache root:", paths.cache_root)


def populate_dataset_cache(
    run_dataset: str,
    downloaded_root: str | Path,
    cache_root: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Copy a KaggleHub dataset into a persistent cache directory.

    The copy is intentionally a plain directory tree so Colab sessions can reuse
    it without relying on KaggleHub's ephemeral VM cache.
    """

    source = Path(downloaded_root)
    target = _cache_dataset_root(run_dataset, cache_root)
    if target.exists() and not overwrite and _is_valid_dataset_root(run_dataset, target):
        return target
    if target.exists() and overwrite:
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)
    return target


def _resolve_cached_dataset(run_dataset: str, cache_root: Path) -> DatasetPaths | None:
    dataset_root = _cache_dataset_root(run_dataset, cache_root)
    if not _is_valid_dataset_root(run_dataset, dataset_root):
        return None
    return DatasetPaths(
        run_dataset=run_dataset,
        dataset_root=dataset_root,
        data_root=_data_root_for(run_dataset, dataset_root),
        output_dir=_output_dir_for(run_dataset),
        source="drive_cache",
        cache_root=cache_root,
    )


def _cache_dataset_root(run_dataset: str, cache_root: str | Path) -> Path:
    if run_dataset not in {"mimic_cxr", "iuxray"}:
        raise ValueError("run_dataset must be one of: mimic_cxr, iuxray")
    return Path(cache_root) / run_dataset


def _data_root_for(run_dataset: str, dataset_root: Path) -> Path:
    if run_dataset == "mimic_cxr":
        official = dataset_root / "official_data_iccv_final"
        return official if official.exists() else dataset_root
    if run_dataset == "iuxray":
        return dataset_root
    raise ValueError("run_dataset must be one of: mimic_cxr, iuxray")


def _output_dir_for(run_dataset: str) -> Path:
    if run_dataset == "mimic_cxr":
        return Path("/content/drive/MyDrive/mimic_dynamic_graph_outputs/flan_t5_small_run1")
    if run_dataset == "iuxray":
        return Path("/content/drive/MyDrive/iuxray_dynamic_graph_outputs/flan_t5_small_run1")
    raise ValueError("run_dataset must be one of: mimic_cxr, iuxray")


def _is_valid_dataset_root(run_dataset: str, dataset_root: Path) -> bool:
    if run_dataset == "iuxray":
        return (dataset_root / "indiana_reports.csv").exists() and (
            dataset_root / "indiana_projections.csv"
        ).exists()
    if run_dataset == "mimic_cxr":
        return (dataset_root / "mimic_cxr_aug_train.csv").exists() and (
            dataset_root / "mimic_cxr_aug_validate.csv"
        ).exists()
    raise ValueError("run_dataset must be one of: mimic_cxr, iuxray")
