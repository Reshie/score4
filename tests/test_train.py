import tempfile
import unittest
from pathlib import Path

try:
    from score4 import train as train_module
except SystemExit:  # pragma: no cover - exercised only when torch is absent.
    train_module = None


@unittest.skipIf(train_module is None, "training helpers require torch")
class TrainOutputTests(unittest.TestCase):
    def test_training_history_round_trips_optional_losses(self) -> None:
        assert train_module is not None
        records = [
            train_module.TrainingRecord(
                iteration=1,
                generated_examples=20,
                buffer_size=20,
                p1_score=-1.0,
                train_steps=0,
                loss=None,
                policy_loss=None,
                value_loss=None,
                elapsed_seconds=1.25,
            ),
            train_module.TrainingRecord(
                iteration=2,
                generated_examples=18,
                buffer_size=38,
                p1_score=1.0,
                train_steps=3,
                loss=2.5,
                policy_loss=1.5,
                value_loss=1.0,
                elapsed_seconds=2.5,
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            train_module._write_training_history(records, path)
            loaded = train_module._load_training_history(path)

        self.assertEqual(len(loaded), 2)
        self.assertIsNone(loaded[0].loss)
        self.assertEqual(loaded[1].iteration, 2)
        self.assertAlmostEqual(loaded[1].loss, 2.5)

    def test_training_chart_writes_svg(self) -> None:
        assert train_module is not None
        records = [
            train_module.TrainingRecord(
                iteration=1,
                generated_examples=20,
                buffer_size=20,
                p1_score=-1.0,
                train_steps=1,
                loss=3.0,
                policy_loss=2.0,
                value_loss=1.0,
                elapsed_seconds=1.0,
            ),
            train_module.TrainingRecord(
                iteration=2,
                generated_examples=22,
                buffer_size=42,
                p1_score=0.0,
                train_steps=1,
                loss=2.4,
                policy_loss=1.7,
                value_loss=0.7,
                elapsed_seconds=1.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_progress.svg"
            train_module._write_training_chart(records, path)
            svg = path.read_text(encoding="utf-8")

        self.assertIn("<svg", svg)
        self.assertIn("Training progress", svg)
        self.assertIn("policy_loss", svg)
        self.assertIn("p1_score", svg)


if __name__ == "__main__":
    unittest.main()
