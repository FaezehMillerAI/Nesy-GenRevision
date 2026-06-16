from __future__ import annotations

import time
from typing import Callable, TypeVar


T = TypeVar("T")


def measure_latency(fn: Callable[[], T], *, warmup: int = 3, repeats: int = 10) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    durations.sort()
    return {
        "mean_ms": 1000.0 * sum(durations) / len(durations),
        "p50_ms": 1000.0 * durations[len(durations) // 2],
        "p95_ms": 1000.0 * durations[min(len(durations) - 1, int(0.95 * len(durations)))],
    }


def parameter_count(model: object) -> int:
    if not hasattr(model, "parameters"):
        return 0
    return int(sum(param.numel() for param in model.parameters() if getattr(param, "requires_grad", False)))

