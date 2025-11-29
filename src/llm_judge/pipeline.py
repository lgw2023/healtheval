from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Sequence

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
        verbose: bool = False,
    ):
        self.loader = CSVDataLoader(data_path)
        self.cache = JSONCache(cache_path) if cache_path else None
        self.prompt_manager = prompt_manager or PromptManager.default_manager()
        self.verbose = verbose
        self.scorer = LLMScorer(
            caller,
            self.prompt_manager,
            cache=self.cache,
            templater=templater,
            verbose=verbose,
        )
        self.report_builder = ReportBuilder()

    def run(
        self,
        configs: Iterable[DecodeConfig],
        repeats: int,
        limit: int | None = None,
        seeds: Sequence[int] | None = None,
    ) -> dict[DecodeConfig, EvaluationReport]:
        config_list = list(configs)
        samples = self.loader.load(limit=limit)
        controller = SamplingController(seeds=seeds)
        grouped_answers: dict[DecodeConfig, List[AnswerScore]] = {cfg: [] for cfg in config_list}

        total_samples = len(samples)
        if self.verbose:
            print(f"[Pipeline] Loaded {total_samples} samples")

        # 记录每个配置已经运行的轮次，便于可视化展示
        run_counters: dict[DecodeConfig, int] = {cfg: 0 for cfg in config_list}

        for seed, config in controller.iter_runs(config_list, repeats=repeats):
            run_counters[config] += 1
            current_round = run_counters[config]
            if self.verbose:
                print(
                    f"[Pipeline] Config prompt={config.prompt_version} "
                    f"temp={config.temperature} top_k={config.top_k} top_p={config.top_p} "
                    f"round={current_round}/{repeats} seed={seed}"
                )

            for idx, sample in enumerate(samples, start=1):
                if self.verbose and (idx == 1 or idx == total_samples or idx % 10 == 0):
                    # 每隔一定间隔打印一次样本进度
                    print(
                        f"[Pipeline]  scoring sample {idx}/{total_samples} "
                        f"(id={sample.sample_id}) for prompt={config.prompt_version}"
                    )

                grouped_answers.setdefault(config, []).extend(
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

        reports: dict[DecodeConfig, EvaluationReport] = {}
        for config, answers in grouped_answers.items():
            if self.verbose:
                print(
                    f"[Pipeline] Building report for prompt={config.prompt_version} "
                    f"temp={config.temperature} top_k={config.top_k} top_p={config.top_p} "
                    f"answers={len(answers)}"
                )
            reports[config] = self.report_builder.summarize(answers)
        if self.cache:
            self.cache.flush()
        return reports
