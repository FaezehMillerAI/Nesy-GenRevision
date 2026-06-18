from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl
from nesy_gen.kg.entity_linking import LexicalEntityLinker
from nesy_gen.kg.primekg import find_primekg_csv


RADIOLOGY_TERMS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "effusion",
    "emphysema",
    "enlarged cardiomediastinum",
    "fracture",
    "infiltrate",
    "lung opacity",
    "mass",
    "mediastinum",
    "nodule",
    "opacity",
    "pneumonia",
    "pneumothorax",
    "pulmonary edema",
    "pleural effusion",
    "right lower lobe",
    "left lower lobe",
    "chest",
    "lung",
    "hilar",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reusable radiology-focused PrimeKG subgraph cache."
    )
    parser.add_argument("--primekg-dir", required=True, help="Full Dataverse PrimeKG folder.")
    parser.add_argument("--manifest", required=True, help="Dataset JSONL manifest for seed extraction.")
    parser.add_argument("--output-dir", required=True, help="Folder where kg.csv and nodes.csv will be saved.")
    parser.add_argument("--hops", type=int, default=1, help="Incident-edge expansion hops around seed nodes.")
    parser.add_argument(
        "--seed-split",
        default="train",
        help="Manifest split allowed to contribute report-derived seed nodes.",
    )
    parser.add_argument("--max-examples", type=int, help="Optional manifest examples to scan for seeds.")
    parser.add_argument("--seed-terms-json", help="Optional JSON list of additional radiology seed terms.")
    args = parser.parse_args()

    primekg_dir = Path(args.primekg_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = primekg_dir / "nodes.csv"
    if not nodes_path.exists():
        raise FileNotFoundError("Radiology subgraph cache requires PrimeKG nodes.csv.")

    kg_csv = find_primekg_csv(primekg_dir)
    if kg_csv is None:
        raise FileNotFoundError(f"No complete PrimeKG CSV found in {primekg_dir}.")

    print(f"Loading nodes: {nodes_path}", flush=True)
    nodes = pd.read_csv(nodes_path)
    vocab = nodes[["node_id", "node_name", "node_type"]].copy()
    vocab["alias"] = vocab["node_name"]
    linker = LexicalEntityLinker(vocab)

    examples = [
        example for example in load_jsonl(args.manifest) if example.split == args.seed_split
    ]
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    seed_terms = list(RADIOLOGY_TERMS)
    if args.seed_terms_json:
        seed_terms.extend(json.loads(Path(args.seed_terms_json).read_text(encoding="utf-8")))

    print("Extracting seed nodes from manifest and radiology term list...", flush=True)
    seed_nodes: set[str] = set()
    for term in seed_terms:
        seed_nodes.update(link.node_id for link in linker.link_text(term))
    for example in tqdm(examples, desc="Manifest seed linking"):
        text = f"{example.indication} {example.report}"
        seed_nodes.update(link.node_id for link in linker.link_text(text))

    print(f"Seed nodes: {len(seed_nodes)}", flush=True)
    print(f"Loading edge list: {kg_csv}", flush=True)
    edges = pd.read_csv(kg_csv, low_memory=False)

    selected_nodes = set(seed_nodes)
    selected_edges = []
    frontier = set(seed_nodes)
    for hop in range(args.hops):
        print(f"Selecting hop {hop + 1}/{args.hops} incident edges...", flush=True)
        mask = edges["x_id"].astype(str).isin(frontier) | edges["y_id"].astype(str).isin(frontier)
        hop_edges = edges.loc[mask].copy()
        selected_edges.append(hop_edges)
        next_nodes = set(hop_edges["x_id"].astype(str)) | set(hop_edges["y_id"].astype(str))
        frontier = next_nodes - selected_nodes
        selected_nodes.update(next_nodes)
        print(
            f"Hop edges={len(hop_edges)}, selected_nodes={len(selected_nodes)}, next_frontier={len(frontier)}",
            flush=True,
        )
        if not frontier:
            break

    if selected_edges:
        sub_edges = pd.concat(selected_edges).drop_duplicates()
    else:
        sub_edges = edges.iloc[0:0].copy()

    node_ids_in_edges = set(sub_edges["x_id"].astype(str)) | set(sub_edges["y_id"].astype(str))
    node_ids = selected_nodes | node_ids_in_edges
    sub_nodes = nodes[nodes["node_id"].astype(str).isin(node_ids)].copy()

    kg_out = output_dir / "kg.csv"
    nodes_out = output_dir / "nodes.csv"
    summary_out = output_dir / "radiology_primekg_summary.json"

    sub_edges.to_csv(kg_out, index=False)
    sub_nodes.to_csv(nodes_out, index=False)
    summary = {
        "source_primekg_dir": str(primekg_dir),
        "source_kg_csv": str(kg_csv),
        "manifest": str(args.manifest),
        "seed_split": args.seed_split,
        "hops": args.hops,
        "manifest_examples_scanned": len(examples),
        "seed_nodes": len(seed_nodes),
        "subgraph_nodes": len(sub_nodes),
        "subgraph_edges": len(sub_edges),
        "kg_csv": str(kg_out),
        "nodes_csv": str(nodes_out),
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
