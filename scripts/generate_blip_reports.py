from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl
from nesy_gen.models.blip_report_generator import (
    BlipReportDataset,
    collate_blip_batch,
    require_blip_dependencies,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reports with a fine-tuned BLIP model.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--prompt", default="a chest x-ray report:")
    args = parser.parse_args()

    deps = require_blip_dependencies()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    BlipProcessor = deps["BlipProcessor"]
    BlipForConditionalGeneration = deps["BlipForConditionalGeneration"]

    examples = [example for example in load_jsonl(args.manifest) if example.split == args.split and example.image_path]
    if args.limit:
        examples = examples[: args.limit]
    processor = BlipProcessor.from_pretrained(args.checkpoint_dir)
    model = BlipForConditionalGeneration.from_pretrained(args.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    loader = DataLoader(
        BlipReportDataset(
            examples,
            processor,
            max_target_length=args.max_new_tokens,
            include_labels=False,
            prompt=args.prompt,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_blip_batch,
    )
    references = {example.study_id: example.report for example in examples}
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="blip generate"):
            generated = model.generate(
                pixel_values=batch["pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
            texts = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for study_id, prediction in zip(batch["study_id"], texts, strict=True):
                rows.append(
                    {
                        "study_id": study_id,
                        "prediction": " ".join(prediction.replace(args.prompt, "").split()),
                        "reference": references[study_id],
                    }
                )
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows)} BLIP predictions to {out}")


if __name__ == "__main__":
    main()

