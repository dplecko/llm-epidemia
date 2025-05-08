# workspace.py  –  minimal loader: “meta” + one config per .parquet file
from __future__ import annotations

import json
import logging
from pathlib import Path

import datasets
import pandas as pd
import numpy as np


# ────────────────────────────────────────────────────────────────────
#  Paths  (put task_specs.json next to this file; adjust BASE if needed)
# ────────────────────────────────────────────────────────────────────
BASE = Path("/Users/patrik/Documents/PhD/repos/llm-epidemia") #Path(__file__).resolve().parent
TASK_SPECS_PATH = BASE / "task_specs.json"


# ────────────────────────────────────────────────────────────────────
#  Logging
# ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
#  Load task_specs.json once
# ────────────────────────────────────────────────────────────────────
with TASK_SPECS_PATH.open() as f:
    TASK_SPECS: list[dict] = json.load(f)

assert TASK_SPECS, "task_specs.json must be a non‑empty list"

for spec in TASK_SPECS:
    # add a helper key that tells us which Parquet each task lives in
    spec["file_key"] = Path(spec["dataset"]).name   # e.g. "acs.parquet"


# ────────────────────────────────────────────────────────────────────
#  Build list of BuilderConfig objects:
#     • one “meta” config   (raw task_specs)
#     • one config per unique .parquet file (10 in your case)
# ────────────────────────────────────────────────────────────────────
def _make_configs() -> list[datasets.BuilderConfig]:
    cfgs = [
        datasets.BuilderConfig(
            name="meta",
            version=datasets.Version("1.0.0"),
            description="Raw metadata (contents of task_specs.json).",
        )
    ]

    seen: set[str] = set()
    for spec in TASK_SPECS:
        key = spec["file_key"]
        if key in seen:
            continue
        seen.add(key)
        cfgs.append(
            datasets.BuilderConfig(
                name=key,
                version=datasets.Version("1.0.0"),
                description=f"Rows from {key}",
            )
        )
    return cfgs


# ────────────────────────────────────────────────────────────────────
#  Dataset builder
# ────────────────────────────────────────────────────────────────────
class Workspace(datasets.GeneratorBasedBuilder):
    """HF dataset with:
       • config “meta”           → task_specs rows
       • one config per .parquet → actual tabular data
    """

    BUILDER_CONFIGS = _make_configs()
    DEFAULT_CONFIG_NAME = "meta"

    # ------------------------------------------------------------------
    # 1) Dataset‑level info
    # ------------------------------------------------------------------
    def _info(self) -> datasets.DatasetInfo:  # noqa: D401
        if self.config.name == "meta":
            # Convert every value to str so Features is rectangular
            feature_dict = {k: datasets.Value("string") for k in TASK_SPECS[0].keys()}
            features = datasets.Features(feature_dict)
        else:
            # Let 🤗 infer the schema directly from the Parquet file
            features = None

        return datasets.DatasetInfo(
            description=self.config.description,
            features=features,
        )

    # ------------------------------------------------------------------
    # 2) Which files belong to this config
    # ------------------------------------------------------------------
    def _split_generators(self, dl_manager):
        if self.config.name == "meta":
            # single pseudo‑split called “meta”
            return [datasets.SplitGenerator(name="meta", gen_kwargs={})]

        parquet_path = BASE / next(
            s for s in TASK_SPECS if s["file_key"] == self.config.name
        )["dataset"]

        # no train/val/test – just a single split called “data”
        return [
            datasets.SplitGenerator(
                name="data", gen_kwargs={"parquet_path": str(parquet_path)}
            )
        ]

    # ------------------------------------------------------------------
    # 3) Row generator
    # ------------------------------------------------------------------
    def _generate_examples(self, parquet_path: str | None = None):
        # 3‑a) meta config – unchanged
        if parquet_path is None:
            for idx, spec in enumerate(TASK_SPECS):
                yield idx, {
                    k: "" if v is None else json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                    for k, v in spec.items()
                }
            return

        # 3‑b) parquet‑backed configs  (✓ fixed NaN/NumPy scalars)
        df = pd.read_parquet(parquet_path)
        for idx, row in df.iterrows():
            record = {}
            for col, val in row.items():
                # turn NaN/NaT into real nulls
                if pd.isna(val):
                    record[col] = None
                else:
                    # unwrap NumPy scalars so Arrow sees plain Python types
                    if isinstance(val, (np.generic,)):
                        record[col] = val.item()
                    else:
                        record[col] = val
            yield idx, record


# Needed for “trust_remote_code=True”
builder_cls = Workspace
