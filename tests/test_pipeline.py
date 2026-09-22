"""CPU-only tests for the evaluation protocol (no model weights or data needed).

    python -m unittest discover tests
"""
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "aegis"))
sys.path.insert(0, str(REPO / "scripts"))

from calibrate_threshold import select_threshold, split_halves  # noqa: E402
from layer_resolve import resolve_layers  # noqa: E402
from refusal import is_refusal  # noqa: E402
from summarize import auroc  # noqa: E402


class RefusalRegexTests(unittest.TestCase):
    def test_refusals(self):
        for text in ["I'm sorry, but I can't help with that.", "I cannot provide that.",
                     "As an AI language model, I won't do this.", "i am unable to assist"]:
            self.assertTrue(is_refusal(text), text)

    def test_only_the_opening_is_checked(self):
        self.assertFalse(is_refusal("Here is how to do it. " * 12 + "I cannot stress this enough."))

    def test_compliance(self):
        for text in ["To kill a Python process, run `kill <pid>`.", "Sure! Here is a recipe.", "", None]:
            self.assertFalse(is_refusal(text), text)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        rng = random.Random(0)
        self.scores = {f"xstest/xs_{i}": rng.random() for i in range(1, 251)}
        self.refused = {k: rng.random() < 0.1 for k in self.scores}

    def test_halves_partition_sorted_ids(self):
        val, test = split_halves(self.scores)
        self.assertEqual(len(val), 125)
        self.assertEqual(len(test), 125)
        self.assertFalse(set(val) & set(test))
        self.assertEqual(sorted(val + test), sorted(self.scores))

    def test_budget_is_respected_on_validation_half(self):
        for budget in (0.0, 0.05, 0.10, 0.5):
            rec = select_threshold(self.scores, self.refused, budget)
            self.assertLessEqual(rec["val_added_over_refusal"], budget)

    def test_threshold_ignores_test_half(self):
        _, test = split_halves(self.scores)
        perturbed = dict(self.scores)
        for k in test:
            perturbed[k] = 1.0 - perturbed[k]
        self.assertEqual(select_threshold(self.scores, self.refused, 0.1)["threshold"],
                         select_threshold(perturbed, self.refused, 0.1)["threshold"])

    def test_baseline_refusals_are_not_counted_as_added(self):
        val, _ = split_halves(self.scores)
        all_refused = {k: True for k in val}
        # every validation prompt was already refused -> nothing can be "added"
        rec = select_threshold(self.scores, all_refused, 0.0)
        self.assertEqual(rec["threshold"], min(self.scores[k] for k in val))


class AurocTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(auroc([0.9, 0.8], [0.1, 0.2]), 1.0)
        self.assertEqual(auroc([0.1], [0.9]), 0.0)
        self.assertEqual(auroc([0.5, 0.5], [0.5, 0.5]), 0.5)

    def test_matches_sklearn(self):
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = random.Random(1)
        pos = [round(rng.random(), 2) for _ in range(200)]
        neg = [round(rng.random() * 0.8, 2) for _ in range(150)]
        expected = roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg)
        self.assertAlmostEqual(auroc(pos, neg), expected, places=10)


class LayerResolveTests(unittest.TestCase):
    def test_absolute_layers(self):
        cfg = {"gate_layer": 21, "late": list(range(25, 42))}
        self.assertEqual(resolve_layers(cfg, 42, 41), (21, list(range(25, 42)), 41))

    def test_out_of_range_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_layers({"gate_layer": 40, "late": [41]}, 32, 31)


class TrainsetTests(unittest.TestCase):
    def test_lobo_and_indomain_folds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            man = tmp / "manifests"
            man.mkdir()
            for name, n in (("audiojailbreak", 30), ("jalm", 30), ("sacred_msd", 30)):
                with open(man / f"{name}.jsonl", "w") as f:
                    for i in range(n):
                        f.write(json.dumps({"id": f"{name}/{i}", "local_audio": f"/a/{name}/{i}.wav",
                                            "prompt": "x"}) + "\n")
            benign = tmp / "benign.jsonl"
            with open(benign, "w") as f:
                for i in range(10):
                    f.write(json.dumps({"id": f"b{i}", "local_audio": f"/a/b/{i}.wav",
                                        "dataset": "jbb_benign" if i < 2 else "benign_helpful",
                                        "m_response": "I'm sorry, but I can't." if i == 0 else "Sure."}) + "\n")
            splits = tmp / "splits"
            splits.mkdir()
            for bench, name in (("audiojailbreak", "audiojailbreak"), ("jalm", "jalm"),
                                ("sacred", "sacred_msd")):
                ids = [f"{name}/{i}" for i in range(30)]
                (splits / f"{bench}_indomain.json").write_text(
                    json.dumps({"train_ids": ids[:15], "test_ids": ids[15:]}))
            out = tmp / "train"
            cmd = [sys.executable, str(REPO / "scripts/build_trainsets.py"), "--manifests", str(man),
                   "--splits", str(splits), "--benign-responses", str(benign),
                   "--response-key", "m_response", "--out-dir", str(out)]
            subprocess.run(cmd, check=True, capture_output=True)
            first = (out / "train_lobo_jalm.jsonl").read_text()
            subprocess.run(cmd, check=True, capture_output=True)
            self.assertEqual(first, (out / "train_lobo_jalm.jsonl").read_text(), "not deterministic")
            for held, folder in (("audiojailbreak", "audiojailbreak"), ("jalm", "jalm"),
                                 ("sacred", "sacred_msd")):
                rows = [json.loads(l) for l in open(out / f"train_lobo_{held}.jsonl")]
                harmful = [r for r in rows if r["kind"] == "harmful"]
                benign_rows = [r for r in rows if r["kind"] == "benign"]
                self.assertEqual(len(benign_rows), 9)          # the refused jbb_benign row is dropped
                self.assertEqual(len(harmful), len(benign_rows))
                self.assertFalse(any(f"/{folder}/" in r["audio"] for r in harmful))
                # in-domain: harmful clips come only from that benchmark's train split
                rows = [json.loads(l) for l in open(out / f"train_indomain_{held}.jsonl")]
                harmful = [r for r in rows if r["kind"] == "harmful"]
                self.assertEqual(len(harmful), 9)
                self.assertTrue(all(f"/{folder}/" in r["audio"] and int(Path(r["audio"]).stem) < 15
                                    for r in harmful))


if __name__ == "__main__":
    unittest.main()
