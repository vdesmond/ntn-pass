"""The analyses behind the README: each study answers one question with the library modules, then writes
results/<study>.json and figures/<study>-*.png."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[3]


class Output:
    """Where a study writes: results/<name>.json and figures/<name>-<figure>.png under `root`."""

    def __init__(self, name, root=ROOT):
        self.name, self.root = name, Path(root)
        (self.root / "figures").mkdir(exist_ok=True)
        (self.root / "results").mkdir(exist_ok=True)

    def figure(self, suffix):
        return self.root / "figures" / f"{self.name}-{suffix}.png"

    def write(self, results):
        (self.root / "results" / f"{self.name}.json").write_text(json.dumps(results, indent=2) + "\n")
