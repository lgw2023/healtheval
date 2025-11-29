from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List

from .cache import JSONCache
from .data_loader import CSVDataLoader, Sample
from .llm_scorer import AnswerScore, LLMScorer
from .prompt_manager import PromptManager
from .reporting import EvaluationReport, ReportBuilder
from .sampling import DecodeConfig, SamplingController
from .templates import InputTemplater


class EvaluationPipeline:
    """High-level pipeline wiring together data loading, scoring, and reporting."""

    def __init__(
        self,
        data_path: Path,
        caller: Callable,
        cache_path: Path | None = None,
        prompt_manager: PromptManager | None = None,
        templater: InputTemplater | None = None,
    ):
        self.loader = CSVDataLoader(data_path)
        self.cache = JSONCache(cache_path) if cache_path else None
        self.prompt_manager = prompt_manager or PromptManager.default_manager()
        self.scorer = LLMScorer(caller, self.prompt_manager, cache=self.cache, templater=templater)
        self.report_builder = ReportBuilder()

    def run(
        self,
        configs: Iterable[DecodeConfig],
        repeats: int,
        limit: int | None = None,
    ) -> dict[DecodeConfig, EvaluationReport]:
        samples = self.loader.load(limit=limit)
        controller = SamplingController()
        reports: dict[DecodeConfig, EvaluationReport] = {}
        for seed, config in controller.iter_runs(configs, repeats=repeats):
            answers: List[AnswerScore] = []
            for sample in samples:
                answers.extend(
                    self.scorer.score_sample(
                        sample,
                        prompt_version=config.prompt_version,
                        temperature=config.temperature,
                        top_k=config.top_k,
                        top_p=config.top_p,
                        run_idx=seed,
                        repeats=1,
                    )
                )
            reports[config] = self.report_builder.summarize(answers)
        if self.cache:
            self.cache.flush()
        return reports
