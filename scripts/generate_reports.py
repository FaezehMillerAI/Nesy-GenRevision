from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl
from nesy_gen.models.report_generator import (
    ReportGenerationDataset,
    collate_generation_batch,
    require_torch_transformers,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reports from a trained image-to-report model.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    deps = require_torch_transformers()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    AutoImageProcessor = deps["AutoImageProcessor"]
    AutoTokenizer = deps["AutoTokenizer"]
    VisionEncoderDecoderModel = deps["VisionEncoderDecoderModel"]

    examples = [example for example in load_jsonl(args.manifest) if example.split == args.split and example.image_path]
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError(f"No examples found for split={args.split}")

    checkpoint = Path(args.checkpoint_dir)
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = ReportGenerationDataset(
        examples,
        image_processor,
        tokenizer,
        max_target_length=args.max_new_tokens,
        include_labels=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_generation_batch,
    )
    rows = []
    references = {example.study_id: example.report for example in examples}
    with torch.no_grad():
        for batch in tqdm(loader, desc="generate"):
            generated = model.generate(
                batch["pixel_values"].to(device),
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for study_id, prediction in zip(batch["study_id"], texts, strict=True):
                rows.append(
                    {
                        "study_id": study_id,
                        "prediction": " ".join(prediction.split()),
                        "reference": references[study_id],
                    }
                )
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows)} predictions to {out}")


if __name__ == "__main__":
    main()

