from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Sequence

from .cache import JSONCache
from .data_loader import CSVDataLoader, Sample
from .llm_scorer import AnswerScore, LLMScorer, Score
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
        # 当 verbose=True 时，也在指标统计阶段输出 task.md 中三类指标的详细计算过程
        self.report_builder = ReportBuilder(debug_metrics=verbose)

    def describe(
        self,
        configs: Iterable[DecodeConfig],
        repeats: int,
        limit: int | None = None,
        seeds: Sequence[int] | None = None,
        combine_weights: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Return a structured view of the current pipeline and run configuration.

        便于在脚本或 Notebook 中以 JSON / dict 的方式查看整体设定，而不是零散的日志。
        """

        config_list = list(configs)
        return {
            "data_path": str(self.loader.path),
            "cache_path": str(getattr(self.cache, "path", "")) if self.cache else None,
            "prompt_versions": [cfg.prompt_version for cfg in config_list],
            "decode_configs": [
                {
                    "prompt_version": cfg.prompt_version,
                    "temperature": cfg.temperature,
                    "top_k": cfg.top_k,
                    "top_p": cfg.top_p,
                }
                for cfg in config_list
            ],
            "repeats": repeats,
            "limit": limit,
            "seeds": list(seeds) if seeds is not None else None,
            "combine_weights": dict(combine_weights) if combine_weights is not None else None,
            "verbose": self.verbose,
        }

    def run(
        self,
        configs: Iterable[DecodeConfig],
        repeats: int,
        limit: int | None = None,
        seeds: Sequence[int] | None = None,
        combine_weights: Mapping[str, float] | None = None,
    ) -> dict[str | DecodeConfig, EvaluationReport]:
        config_list = list(configs)
        samples = self.loader.load(limit=limit)
        controller = SamplingController(seeds=seeds)
        grouped_answers: dict[DecodeConfig, List[AnswerScore]] = {cfg: [] for cfg in config_list}

        total_samples = len(samples)
        if self.verbose:
            print(f"[Pipeline] Loaded {total_samples} samples")

        # 记录每个配置已经运行的轮次，便于可视化展示
        run_counters: dict[DecodeConfig, int] = {cfg: 0 for cfg in config_list}

        # 这里我们将 repeats 的语义专门用于“单条样本多次打分”（Krippendorff α 所需），
        # 因此外层按配置/seed 的迭代固定为 1 轮，以避免不必要的重复调用和覆盖。
        for seed, config in controller.iter_runs(config_list, repeats=1):
            run_counters[config] += 1
            current_round = run_counters[config]
            if self.verbose:
                print(
                    f"[Pipeline] Config prompt={config.prompt_version} "
                    f"temp={config.temperature} top_k={config.top_k} top_p={config.top_p} "
                    f"round={current_round}/1 seed={seed}"
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
                        # repeats 用于同一 (sample, answer) 的多次打分，从而支撑单条打分稳定性 (α)
                        repeats=repeats,
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

        # If weights are provided (defaulting to ground:structure = 2:1),
        # aggregate the prompt-specific answers into a single weighted report.
        weight_map = combine_weights or {"ground": 2.0, "structure": 1.0}
        if weight_map:
            combined_answers = self._combine_weighted_answers(grouped_answers, weight_map)
            # 单条打分稳定性仍然基于原始的逐 prompt 打分结果，成对比较则使用权重加和后的总分
            stability_answers: List[AnswerScore] = []
            for cfg, answers in grouped_answers.items():
                if cfg.prompt_version in weight_map:
                    stability_answers.extend(answers)

            if self.verbose:
                print(
                    f"[Pipeline] Building weighted combined report using weights: {weight_map} "
                    f"answers={len(combined_answers)}"
                )
            reports["weighted_combined"] = self.report_builder.summarize_from_separate_sources(
                stability_source=stability_answers, decision_source=combined_answers
            )
        if self.cache:
            self.cache.flush()
        return reports

    @staticmethod
    def _combine_weighted_answers(
        grouped_answers: dict[DecodeConfig, List[AnswerScore]],
        weights: Mapping[str, float],
    ) -> List[AnswerScore]:
        """Merge answers from multiple prompt versions via weighted averaging.

        The weighted score is computed as::

            sum(weight[prompt] * score) / sum(weight)

        Only prompts present in ``weights`` participate in the aggregation.
        """

        combined: dict[tuple[str, str, int], dict[str, AnswerScore]] = {}
        for answers in grouped_answers.values():
            for ans in answers:
                key = (ans.sample_id, ans.answer_id, ans.run_idx)
                combined.setdefault(key, {})[ans.prompt_version] = ans

        merged_scores: List[AnswerScore] = []
        for _, prompt_answers in combined.items():
            weighted_total = 0.0
            weighted_confidence = 0.0
            total_weight = 0.0

            for prompt, ans in prompt_answers.items():
                weight = weights.get(prompt, 0.0)
                if weight <= 0 or ans.parsed is None:
                    continue
                avg_score = ans.parsed.average_score
                weighted_total += weight * avg_score
                weighted_confidence += weight * (ans.parsed.confidence or 0.0)
                total_weight += weight

            if total_weight == 0:
                continue

            final_score = weighted_total / total_weight
            final_confidence = weighted_confidence / total_weight if total_weight else 0.0

            # Reuse decode params from any constituent answer
            any_answer = next(iter(prompt_answers.values()))
            merged_scores.append(
                AnswerScore(
                    sample_id=any_answer.sample_id,
                    answer_id=any_answer.answer_id,
                    human_winner=any_answer.human_winner,
                    prompt_version="weighted_combined",
                    decode_params=any_answer.decode_params,
                    raw_response=(
                        "weighted combination: "
                        f"{', '.join(f'{p}={weights.get(p, 0)}' for p in prompt_answers)}"
                    ),
                    parsed=Score(checks=[{"score": final_score}], confidence=final_confidence),
                    run_idx=any_answer.run_idx,
                )
            )

        return merged_scores
