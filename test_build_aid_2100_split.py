import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from build_aid_2100_split import build_split


class BuildAidSplitTest(unittest.TestCase):
    def make_source(self, root: Path) -> Path:
        source = root / "source"
        for class_index, class_name in enumerate(("alpha", "beta", "gamma")):
            class_dir = source / class_name
            class_dir.mkdir(parents=True)
            for image_index in range(6):
                Image.new(
                    "RGB",
                    (12, 12),
                    color=(class_index * 60, image_index * 30, image_index),
                ).save(class_dir / f"{image_index}.png")
        return source

    def test_balanced_reproducible_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            summaries = []
            manifests = []
            for name in ("first", "second"):
                output = root / name
                summaries.append(
                    build_split(
                        source,
                        output,
                        seed=2026,
                        expected_classes=3,
                        samples_per_class=5,
                        train_per_class=3,
                        materialize_mode="copy",
                        test_per_class=1,
                    )
                )
                with (output / "split_manifest.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    manifests.append(list(csv.DictReader(handle)))

            self.assertEqual(summaries[0]["num_selected_images"], 15)
            self.assertEqual(summaries[0]["num_train_images"], 9)
            self.assertEqual(summaries[0]["num_val_images"], 3)
            self.assertEqual(summaries[0]["num_test_images"], 3)
            self.assertEqual(summaries[0]["num_cross_split_duplicate_groups"], 0)
            self.assertEqual(summaries[0]["class_counts"]["train"], {
                "alpha": 3,
                "beta": 3,
                "gamma": 3,
            })
            self.assertEqual(
                [row["source_relative_path"] for row in manifests[0]],
                [row["source_relative_path"] for row in manifests[1]],
            )
            self.assertEqual(len(list((root / "first" / "train_hr").iterdir())), 9)
            self.assertEqual(len(list((root / "first" / "val_hr").iterdir())), 3)
            self.assertEqual(len(list((root / "first" / "test_hr").iterdir())), 3)
            metadata = json.loads(
                (root / "first" / "split_manifest_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["dataset"], "AID-15")


if __name__ == "__main__":
    unittest.main()
