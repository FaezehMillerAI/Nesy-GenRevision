from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    run_dataset: str
    dataset_root: Path
    data_root: Path
    output_dir: Path


def resolve_kaggle_dataset(run_dataset: str) -> DatasetPaths:
    """Resolve the Colab/KaggleHub dataset layout used for the revision runs."""

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

    return DatasetPaths(
        run_dataset=run_dataset,
        dataset_root=dataset_root,
        data_root=data_root,
        output_dir=output_dir,
    )


def print_dataset_paths(paths: DatasetPaths) -> None:
    print("Dataset:", paths.run_dataset)
    print("Dataset root:", paths.dataset_root)
    print("Data root:", paths.data_root)
    print("Exists:", paths.data_root.exists())
    print("Output dir:", paths.output_dir)

