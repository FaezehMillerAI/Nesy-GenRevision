from __future__ import annotations

import random
from collections.abc import Sequence

from nesy_gen.evaluation.metrics import entity_f1


def run_linking_sensitivity(
    reference_node_ids: Sequence[str],
    *,
    candidate_node_ids: Sequence[str],
    drop_rates: Sequence[float],
    swap_rates: Sequence[float],
    trials: int = 100,
    seed: int = 13,
) -> list[dict[str, float]]:
    rng = random.Random(seed)
    rows: list[dict[str, float]] = []
    reference = [str(node_id) for node_id in reference_node_ids]
    candidates = [str(node_id) for node_id in candidate_node_ids]

    for drop_rate in drop_rates:
        for swap_rate in swap_rates:
            f1_values = []
            for _ in range(trials):
                perturbed = []
                for node_id in reference:
                    if rng.random() < drop_rate:
                        continue
                    if candidates and rng.random() < swap_rate:
                        perturbed.append(rng.choice(candidates))
                    else:
                        perturbed.append(node_id)
                f1_values.append(entity_f1(perturbed, reference)["f1"])
            rows.append(
                {
                    "drop_rate": float(drop_rate),
                    "swap_rate": float(swap_rate),
                    "mean_f1": sum(f1_values) / len(f1_values),
                    "trials": float(trials),
                }
            )
    return rows

