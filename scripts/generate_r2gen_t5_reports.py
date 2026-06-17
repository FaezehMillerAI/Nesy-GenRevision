from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl
from nesy_gen.models.r2gen_t5 import (
    R2GenT5Dataset,
    R2GenT5Model,
    collate_r2gen_t5_batch,
    decode_r2gen_predictions,
    require_r2gen_t5_dependencies,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reports with the Vision-T5 image-to-report model.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--num-beam-groups", type=int, default=1)
    parser.add_argument("--diversity-penalty", type=float, default=0.0)
    args = parser.parse_args()

    deps = require_r2gen_t5_dependencies()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]

    model = R2GenT5Model.from_pretrained(args.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    examples = [example for example in load_jsonl(args.manifest) if example.split == args.split and example.image_path]
    if args.limit:
        examples = examples[: args.limit]
    loader = DataLoader(
        R2GenT5Dataset(
            examples,
            model.tokenizer,
            max_target_length=model.config.max_target_length,
            include_labels=False,
            target_prefix=model.config.target_prefix,
            image_size=model.config.image_size,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_r2gen_t5_batch,
    )
    references = {example.study_id: example.report for example in examples}
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Vision-T5 generate"):
            generated = model.generate(
                batch["image"].to(device),
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                do_sample=args.do_sample,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                length_penalty=args.length_penalty,
                num_beam_groups=args.num_beam_groups,
                diversity_penalty=args.diversity_penalty,
            )
            texts = decode_r2gen_predictions(
                model.tokenizer,
                generated,
                target_prefix=model.config.target_prefix,
            )
            for idx, study_id in enumerate(batch["study_id"]):
                prediction = texts[idx * args.num_return_sequences]
                rows.append(
                    {
                        "study_id": study_id,
                        "prediction": prediction,
                        "reference": references[study_id],
                    }
                )

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows)} Vision-T5 predictions to {out}")


if __name__ == "__main__":
    main()
