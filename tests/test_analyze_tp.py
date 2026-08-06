import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from analyze_tp import compare  # noqa: E402


class AnalyzeTpTest(unittest.TestCase):
    def write_run(self, root, name, multiplier):
        rows = [
            {"scenario": "latency", "concurrency": 1,
             "per_request_decode_tok_s": 100 * multiplier,
             "mean_output_tokens": 32},
            {"scenario": "throughput", "concurrency": 2,
             "output_tok_s": 200 * multiplier, "mean_output_tokens": 32},
            {"scenario": "prefill", "concurrency": 1,
             "target_prompt_tokens": 512, "prefill_tok_s": 300 * multiplier},
        ]
        path = root / f"{name}.json"
        path.write_text(json.dumps({"model": "test", "rows": rows}))
        return path

    def test_compares_matched_rows_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = compare(self.write_run(root, "tp1", 0.5),
                             self.write_run(root, "tp2", 1.0),
                             self.write_run(root, "scraped", 0.995),
                             self.write_run(root, "sampled", 0.98), 1.0)
        self.assertTrue(result["overhead_budget_pass"])
        self.assertAlmostEqual(result["scrape_overhead_percent"], 0.5)
        self.assertEqual(len(result["rows"]), 3)

    def test_rejects_a_different_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [self.write_run(root, name, 1.0)
                     for name in ("tp1", "tp2", "scraped", "sampled")]
            doc = json.loads(paths[2].read_text())
            doc["rows"].pop()
            paths[2].write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "matrix"):
                compare(*paths, 1.0)

    def test_fails_the_overhead_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = compare(self.write_run(root, "tp1", 1.0),
                             self.write_run(root, "tp2", 1.0),
                             self.write_run(root, "scraped", 0.97),
                             self.write_run(root, "sampled", 1.0), 1.0)
        self.assertFalse(result["overhead_budget_pass"])
        self.assertAlmostEqual(result["scrape_overhead_percent"], 3.0)


if __name__ == "__main__":
    unittest.main()
