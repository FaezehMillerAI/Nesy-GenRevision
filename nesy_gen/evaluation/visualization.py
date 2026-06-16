from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from nesy_gen.data.schema import RadiologyExample


def build_qualitative_html(
    *,
    examples: list[RadiologyExample],
    predictions: pd.DataFrame,
    run_name: str,
    graph_scores: pd.DataFrame | None = None,
    retrieval: pd.DataFrame | None = None,
    factuality: pd.DataFrame | None = None,
    max_examples: int = 25,
) -> str:
    examples_by_id = {example.study_id: example for example in examples}
    graph_by_id = _indexed_records(graph_scores)
    retrieval_by_id = _indexed_records(retrieval)
    factuality_by_id = _indexed_records(factuality)

    rows = []
    for row in predictions.head(max_examples).itertuples(index=False):
        study_id = str(getattr(row, "study_id"))
        example = examples_by_id.get(study_id)
        if example is None:
            continue
        score = graph_by_id.get(study_id, {})
        retrieved = retrieval_by_id.get(study_id, {})
        fact = factuality_by_id.get(study_id, {})
        rows.append(
            _render_example(
                study_id=study_id,
                image_path=example.image_path or "",
                prediction=str(getattr(row, "prediction")),
                reference=str(getattr(row, "reference", example.report)),
                retrieved=str(retrieved.get("prediction", "")),
                score=score,
                factuality=fact,
            )
        )

    return f"""<html>
<head>
  <meta charset="utf-8">
  <title>{escape(run_name)} Qualitative Report</title>
</head>
<body style="font-family:Arial, sans-serif; margin:24px; color:#202124;">
  <h1>{escape(run_name)} Qualitative Report</h1>
  <p>Examples: {len(rows)}</p>
  {''.join(rows)}
</body>
</html>
"""


def save_qualitative_html(html: str, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def save_standard_plots(
    *,
    output_dir: str | Path,
    run_name: str,
    graph_scores: pd.DataFrame | None = None,
    factuality: pd.DataFrame | None = None,
    sensitivity: pd.DataFrame | None = None,
    entities: pd.DataFrame | None = None,
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not any(frame is not None for frame in (graph_scores, factuality, sensitivity, entities)):
        return paths
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install matplotlib to create result plots.") from exc

    if graph_scores is not None and "mean" in graph_scores:
        path = out / f"{run_name}_graph_score_hist.png"
        plt.figure(figsize=(7, 4))
        graph_scores["mean"].hist(bins=20)
        plt.title("Graph Consistency Score Distribution")
        plt.xlabel("Mean graph consistency")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)

    if factuality is not None and "hallucination_rate" in factuality:
        path = out / f"{run_name}_hallucination_hist.png"
        plt.figure(figsize=(7, 4))
        factuality["hallucination_rate"].hist(bins=20)
        plt.title("Generated Report Entity Hallucination Rate")
        plt.xlabel("Entity hallucination rate")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)

    if sensitivity is not None and {"drop_rate", "swap_rate", "mean_f1"}.issubset(sensitivity.columns):
        path = out / f"{run_name}_sensitivity_curve.png"
        summary = sensitivity.groupby(["drop_rate", "swap_rate"])["mean_f1"].mean().reset_index()
        plt.figure(figsize=(7, 4))
        for swap_rate, group in summary.groupby("swap_rate"):
            plt.plot(group["drop_rate"], group["mean_f1"], marker="o", label=f"swap={swap_rate}")
        plt.title("Entity-Linking Sensitivity")
        plt.xlabel("Entity drop rate")
        plt.ylabel("Mean entity F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)

    if entities is not None and {"node_name", "count"}.issubset(entities.columns) and not entities.empty:
        path = out / f"{run_name}_top_positive_entities.png"
        top = entities.head(15)
        plt.figure(figsize=(8, 5))
        plt.barh(top["node_name"][::-1], top["count"][::-1])
        plt.title("Top Positive Linked Entities")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)

    return paths


def _indexed_records(frame: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if frame is None or frame.empty or "study_id" not in frame.columns:
        return {}
    return frame.set_index("study_id").to_dict(orient="index")


def _render_example(
    *,
    study_id: str,
    image_path: str,
    prediction: str,
    reference: str,
    retrieved: str,
    score: dict[str, object],
    factuality: dict[str, object],
) -> str:
    return f"""
    <div style="border:1px solid #dadce0; padding:14px; margin:14px 0; border-radius:8px;">
      <h2 style="margin-top:0;">{escape(study_id)}</h2>
      <div style="display:flex; gap:16px; align-items:flex-start;">
        <div>
          <img src="{escape(image_path)}" style="max-width:280px; max-height:280px; border:1px solid #ccc;">
        </div>
        <div style="max-width:900px;">
          <p><b>Graph scores:</b>
            mean={escape(str(score.get("mean", "NA")))},
            bio={escape(str(score.get("bio_temporal", "NA")))},
            finding-dx={escape(str(score.get("finding_to_diagnosis", "NA")))},
            located-in={escape(str(score.get("located_in_type", "NA")))}
          </p>
          <p><b>Generated factuality:</b>
            F1={escape(str(factuality.get("f1", "NA")))},
            hallucination={escape(str(factuality.get("hallucination_rate", "NA")))},
            negation mismatches={escape(str(factuality.get("negation_mismatch_count", "NA")))}
          </p>
          <p><b>Generated:</b><br>{escape(prediction)}</p>
          <p><b>Reference:</b><br>{escape(reference)}</p>
          <p><b>Retrieval baseline:</b><br>{escape(retrieved)}</p>
        </div>
      </div>
    </div>
    """
