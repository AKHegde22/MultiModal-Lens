"""Evaluation utilities for batch analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from multimodallens.core.pipeline import LensPipeline


@dataclass(slots=True)
class EvalRecord:
    sample_id: str
    image_path: Path
    prompt: str


class EvaluationRunner:
    """Batch evaluator for local JSONL datasets."""

    def __init__(self, pipeline: LensPipeline | None = None) -> None:
        self.pipeline = pipeline or LensPipeline()

    @staticmethod
    def load_jsonl(path: str | Path) -> list[EvalRecord]:
        records: list[EvalRecord] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                image_path = Path(row["image"]).expanduser()
                sample_id = str(row.get("id", idx))
                prompt = str(row["text"])
                records.append(EvalRecord(sample_id=sample_id, image_path=image_path, prompt=prompt))
        return records

    def run(
        self,
        dataset_path: str | Path,
        family: str,
        model_name: str,
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        compute_gradients: bool = False,
        run_faithfulness: bool = False,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        data = self.load_jsonl(dataset_path)

        for item in data:
            image = Image.open(item.image_path).convert("RGB")
            result = self.pipeline.analyze(
                family=family,
                model_name=model_name,
                image=image,
                prompt=item.prompt,
                device=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                compute_gradients=compute_gradients,
                run_faithfulness=run_faithfulness,
            )

            row: dict[str, object] = {
                "id": item.sample_id,
                "image": str(item.image_path),
                "prompt": item.prompt,
                "global_score": result.global_score,
                "num_tokens": len(result.tokens),
                "num_patches": result.patch_grid[0] * result.patch_grid[1],
            }

            if result.faithfulness is not None:
                row["counterfactual_drop"] = result.faithfulness.counterfactual_drop
                row["attn_grad_spearman"] = result.faithfulness.attn_grad_spearman
                row["deletion_auc"] = float(sum(result.faithfulness.deletion_curve))
                row["insertion_auc"] = float(sum(result.faithfulness.insertion_curve))

            rows.append(row)

        return pd.DataFrame(rows)
