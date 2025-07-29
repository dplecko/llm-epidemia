
import evaluate
import datasets
from huggingface_hub import cached_assets_path
from transformers import PreTrainedModel
from typing import List, Dict, Any
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from eval import build_eval_df
from extract import task_extract


class LLMObservatoryEval(evaluate.Metric):
    """HuggingFace Evaluate wrapper so users can do `evaluate.load(...)`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from task_spec import task_specs, task_specs_hd
        self.task_specs = task_specs
        self.task_specs_hd = task_specs_hd

    def _info(self) -> evaluate.MetricInfo:  # type: ignore[override]
        return evaluate.MetricInfo(
            description=(
                "Benchmark score used by the LLM Observatory. "
                "Given a list of model names and a list of task "
                "dictionaries, computes per-task and overall scores "
                "on categorical and high-dimensional prediction tasks."
            ),
            citation="",
            inputs_description=(
                "• **models** (list of str): model identifiers.\n"
                "• **tasks**  (list of dicts): task specifications "
                "(see repository README).\n"
                "• **prob**   (bool, optional): whether predictions are "
                "probabilities (default False)."
            ),
            features=datasets.Features({}),               # free-form arguments → no Features()
            reference_urls=[
                "https://github.com/dplecko/llm-epidemia"
            ],
        )
        
    def compute(                 
        self,
        *,                       # force keyword-only, like HF’s own metrics
        models: List[str],
        tasks:  List[Dict[str, Any]],
        prob:   bool = False,
        **ignored,               
    ):
        """Compute benchmark scores for LLM-Observatory tasks."""
        return self._compute(models=models, tasks=tasks, prob=prob)

    def _compute(   # type: ignore[override]
        self,
        models: List[str],
        tasks:  List[Dict[str, Any]],
        prob:   bool = False,
    ) -> Dict[str, Any]:
        cache_dir = cached_assets_path("llm-observatory", namespace="default", subfolder="data/benchmark")
        df, _ = build_eval_df(models, tasks, prob=prob, cache_dir=cache_dir)
        return {
            "overall_score": float(df["score"].mean()),
            "per_task":      df.to_dict(orient="records"),
        }
    
    def extract(                 
        self,
        *,                       # force keyword-only, like HF’s own metrics
        model_name: str,
        model: PreTrainedModel,
        task:  Dict[str, Any],
        prob:   bool = False,
        **ignored,               
    ):
        """Compute benchmark scores for LLM-Observatory tasks."""
        return self._extract(model_name=model_name, model=model, task=task, prob=prob)

    def _extract(   # type: ignore[override]
        self,
        model_name: str,
        model: PreTrainedModel,
        task:  Dict[str, Any],
        prob:   bool = False,
    ) -> Dict[str, Any]:
        cache_dir = cached_assets_path("llm-observatory", namespace="default", subfolder="data/benchmark")
        task_extract(model_name, model, task, check_cache=True, prob=prob, cache_dir=cache_dir)
        return str(cache_dir)
