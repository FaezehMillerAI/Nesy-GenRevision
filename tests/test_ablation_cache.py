import sys
import unittest
from types import SimpleNamespace

from scripts.run_ablation_suite import build_commands


class AblationCacheTest(unittest.TestCase):
    def test_compatible_variants_share_candidate_cache(self):
        args = SimpleNamespace(
            manifest="manifest.jsonl",
            primekg_dir="primekg",
            output_dir="outputs",
            dataset_name="iuxray",
            generator_checkpoint_dir="checkpoint",
            retrieval_mode="visual",
            split="test",
            limit=25,
            batch_size=2,
            max_new_tokens=140,
        )

        commands = build_commands(args)
        generation = [
            item
            for item in commands
            if item["cmd"]
            and item["cmd"][0] == sys.executable
            and "scripts/generate_rag_primekg_reports.py" in item["cmd"]
        ]
        writers = [item for item in generation if "--candidate-cache-out" in item["cmd"]]
        readers = [item for item in generation if "--candidate-cache-in" in item["cmd"]]

        self.assertEqual(len(writers), 3)
        self.assertEqual(len(readers), 5)
        graph_top10 = "outputs/iuxray_candidate_cache_graph_top10.json"
        self.assertEqual(sum(graph_top10 in item["cmd"] for item in writers), 1)
        self.assertEqual(sum(graph_top10 in item["cmd"] for item in readers), 5)


if __name__ == "__main__":
    unittest.main()
