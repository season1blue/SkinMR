import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_runner", ROOT / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReleaseTests(unittest.TestCase):
    def test_source_integrity(self):
        runner.verify_sources()
        snapshot = runner.read_json(ROOT / "provenance/source_snapshot.json")
        self.assertTrue(all(x["file"].startswith("code/") for x in snapshot["files"]))

    def test_commands_are_self_contained(self):
        profiles = runner.read_json(ROOT / "configs/profiles.json")
        self.assertTrue(profiles)
        for profile in profiles.values():
            code, commands = runner.commands(profile, Path("model"), Path("data.csv"), Path("images"), Path("output"))
            self.assertEqual(code, ROOT / "code")
            self.assertTrue(Path(commands[0][2]).is_file())
            self.assertTrue(Path(commands[0][2]).is_relative_to(ROOT))
            if "mode" in profile:
                self.assertEqual("--use_cd" in commands[0], profile["mode"] != "base")
                self.assertEqual(len(commands), 2)
            if "batch_size" in profile:
                self.assertEqual(commands[0][commands[0].index("--batch-size") + 1], str(profile["batch_size"]))
                self.assertNotIn("--do-sample", commands[0])

    def test_profile_data_dependencies(self):
        profiles = runner.read_json(ROOT / "configs/profiles.json")
        datasets = runner.read_json(ROOT / "configs/datasets.json")
        self.assertEqual({x["dataset"] for x in profiles.values()}, set(datasets))
        for item in datasets.values():
            self.assertGreater(item["n"], 0)
            self.assertEqual(len(item["csv_sha256"]), 64)

    def test_aggregate_counts(self):
        with (ROOT / "results/summary.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows)
        names = {x["profile"] for x in rows}
        profiles = runner.read_json(ROOT / "configs/profiles.json")
        self.assertTrue(set(profiles).issubset(names))
        for row in rows:
            self.assertAlmostEqual(int(row["correct"]) / int(row["n"]), float(row["accuracy"]), places=12)

    def test_input_validation(self):
        private = ROOT / "runs"
        private.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=private) as directory:
            root = Path(directory)
            image = root / "synthetic.txt"
            image.write_text("Path validation only; not used for inference")
            data = root / "synthetic.csv"
            with data.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image", "categories"])
                writer.writeheader()
                writer.writerow(dict(image=image.name, categories="['tumor']"))
            expected = dict(n=1, labels_observed_in_csv_order=["tumor"], csv_sha256=runner.sha256(data))
            self.assertEqual(runner.validate_data(data, root, expected)[0], 1)
            with self.assertRaises(ValueError):
                runner.validate_data(data, root, dict(expected, csv_sha256="invalid"))
            image.unlink()
            with self.assertRaises(FileNotFoundError):
                runner.validate_data(data, root, expected)


if __name__ == "__main__":
    unittest.main()
