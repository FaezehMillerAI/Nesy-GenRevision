from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class RadiologyExample:
    """Dataset-neutral representation for IU X-ray and MIMIC-CXR rows."""

    study_id: str
    image_path: str | None
    indication: str
    report: str
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)


def load_jsonl(path: str | Path) -> list[RadiologyExample]:
    examples: list[RadiologyExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(RadiologyExample(**row))
    return examples


def write_jsonl(path: str | Path, examples: Iterable[RadiologyExample]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")

