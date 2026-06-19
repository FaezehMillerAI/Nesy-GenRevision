from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a readable adaptive NeSy explanation report.")
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    with Path(args.trace_jsonl).open(encoding="utf-8") as handle:
        traces = [json.loads(line) for line in handle if line.strip()][: args.limit]
    output = Path(args.output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render(traces), encoding="utf-8")
    print(f"Explanation report: {output}")


def _render(traces: list[dict[str, object]]) -> str:
    studies = "".join(_study(trace) for trace in traces)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Adaptive NeSy-Gen Explanations</title><style>
body{{font:15px/1.5 system-ui,sans-serif;margin:0;color:#17202a;background:#f6f7f9}}
main{{max-width:1180px;margin:auto;padding:28px}} h1,h2{{margin:0 0 12px}}
.study{{background:white;border:1px solid #dfe3e8;margin:18px 0;padding:18px;border-radius:6px}}
.claim{{border-top:1px solid #e8eaed;padding:14px 0}} .claim:first-of-type{{border-top:0}}
.row{{display:grid;grid-template-columns:160px 1fr;gap:12px;margin:5px 0}}
.label{{font-weight:650;color:#52606d}} .decision{{font-weight:700}}
.accept_fast_path,.accept_verified{{color:#08783e}} .revise{{color:#986400}} .flag,.abstain{{color:#a82d2d}}
.metric{{display:inline-block;margin:3px 8px 3px 0;padding:2px 7px;background:#eef1f4;border-radius:4px}}
@media(max-width:700px){{.row{{grid-template-columns:1fr;gap:1px}}main{{padding:14px}}}}
</style></head><body><main><h1>Adaptive Neuro-Symbolic Evidence Traces</h1>
<p>These records expose evidence and decisions used during inference; they are not post-hoc LLM rationales.</p>
{studies}</main></body></html>"""


def _study(trace: dict[str, object]) -> str:
    claims = "".join(_claim(claim) for claim in trace.get("claims", []))
    return f"""<section class="study"><h2>{escape(str(trace.get('study_id', 'Study')))}</h2>
<div class="row"><span class="label">Original report</span><span>{escape(str(trace.get('original_report', '')))}</span></div>
<div class="row"><span class="label">Final report</span><span>{escape(str(trace.get('final_report', '')))}</span></div>
{claims}</section>"""


def _claim(claim: dict[str, object]) -> str:
    decision = str(claim.get("decision", ""))
    entities = ", ".join(
        f"{entity.get('node_name', '')} ({'negated' if entity.get('negated') else 'positive'})"
        for entity in claim.get("linked_entities", [])
    ) or "No linked clinical entity"
    path = " → ".join(str(node.get("node_name", node.get("node_id", ""))) for node in claim.get("primekg_path", [])) or "Not invoked / unavailable"
    metrics = "".join(
        f'<span class="metric">{label}: {float(claim.get(key, 0.0)):.3f}</span>'
        for key, label in [
            ("visual_support", "visual"),
            ("retrieval_support", "retrieval"),
            ("gate_confidence", "gate"),
        ]
    )
    return f"""<article class="claim"><div class="row"><span class="label">Claim</span><span>{escape(str(claim.get('original_claim', '')))}</span></div>
<div class="row"><span class="label">Entities</span><span>{escape(entities)}</span></div>
<div class="row"><span class="label">Evidence</span><span>{metrics}</span></div>
<div class="row"><span class="label">PrimeKG path</span><span>{escape(path)}</span></div>
<div class="row"><span class="label">Decision</span><span class="decision {escape(decision)}">{escape(decision)}: {escape(str(claim.get('reason', '')))}</span></div>
<div class="row"><span class="label">Final claim</span><span>{escape(str(claim.get('final_claim', '')))}</span></div></article>"""


if __name__ == "__main__":
    main()
