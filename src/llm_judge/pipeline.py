from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        *,
        prompt_manager: PromptManager | None = None,
        templater: InputTemplater | None = None,
        verbose: bool = False,
        max_workers: int | None = None,
        # 传递给 LLMScorer 的缺失维度处理策略，含义同 LLMScorer.__init__。
        missing_dims_strategy: str = "fill_max",
        missing_full_score: float = 5.0,
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
            missing_dims_strategy=missing_dims_strategy,
            missing_full_score=missing_full_score,
        )
        # 控制并发调用大模型的线程数；为 None 或 <=1 时退回到串行模式
        self.max_workers = max_workers if (max_workers or 0) > 1 else None
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

            # 如果设置了 max_workers，则对单个配置内的样本进行并发打分；
            # 否则退回到原来的串行逻辑，保持完全兼容。
            if self.max_workers:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_meta = {}
                    for idx, sample in enumerate(samples, start=1):
                        future = executor.submit(
                            self.scorer.score_sample,
                            sample,
                            config.prompt_version,
                            config.temperature,
                            config.top_k,
                            config.top_p,
                            seed,
                            # repeats 用于同一 (sample, answer) 的多次打分，从而支撑单条打分稳定性 (α)
                            repeats,
                        )
                        future_to_meta[future] = (idx, sample.sample_id)

                    for future in as_completed(future_to_meta):
                        idx, sample_id = future_to_meta[future]
                        if self.verbose and (idx == 1 or idx == total_samples or idx % 10 == 0):
                            print(
                                f"[Pipeline]  scoring sample {idx}/{total_samples} "
                                f"(id={sample_id}) for prompt={config.prompt_version}"
                            )
                        answers = future.result()
                        grouped_answers.setdefault(config, []).extend(answers)
            else:
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
        weight_map = combine_weights or {"ground": 2.0, "structure": 1.0}

        # 为 Krippendorff’s α 准备「原始评分集合」：
        # 直接使用各个 prompt 版本（如 GROUND_PROMPT_TPL / STRUCT_PROMPT_TPL）
        # 的逐规则打分结果，而不是加权融合后的总分。
        raw_answers_for_alpha: List[AnswerScore] = []
        for cfg_answers in grouped_answers.values():
            raw_answers_for_alpha.extend(cfg_answers)

        combined_answers, weight_debug = self._combine_weighted_answers(
            grouped_answers,
            weight_map,
            debug=self.verbose,
        )
        if self.verbose:
            print(
                f"[Pipeline] Building weighted combined report using weights: {weight_map} "
                f"answers={len(combined_answers)}"
            )
        # 注意：
        # - α 基于原始 prompt 的逐规则打分；
        # - pair-accuracy / Alt-Test 仍基于加权结果；
        # - 为了便于阅读，将「加权融合：原始 prompt 得分与权重贡献」的可视化日志
        #   放在 Krippendorff α 之后统一打印，因此这里将 weight_debug 传递给 ReportBuilder。
        reports["weighted_combined"] = self.report_builder.summarize(
            combined_answers,
            answers_for_alpha=raw_answers_for_alpha,
            weight_debug=weight_debug,
        )
        if self.cache:
            self.cache.flush()
        return reports

    @staticmethod
    def _combine_weighted_answers(
        grouped_answers: dict[DecodeConfig, List[AnswerScore]],
        weights: Mapping[str, float],
        debug: bool = False,
    ) -> tuple[List[AnswerScore], list] | tuple[List[AnswerScore], None]:
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
        # 收集用于后续统一打印的加权融合中间结果（仅在 debug=True 时使用）
        debug_info: list | None = [] if debug else None
        for _, prompt_answers in combined.items():
            weighted_total = 0.0
            weighted_confidence = 0.0
            total_weight = 0.0

            # 为可视化加权过程准备中间日志
            debug_rows: List[tuple[str, float, float, float]] = [] if debug else None

            for prompt, ans in prompt_answers.items():
                weight = weights.get(prompt, 0.0)
                if weight <= 0 or ans.parsed is None:
                    continue
                avg_score = ans.parsed.average_score
                weighted_total += weight * avg_score
                weighted_confidence += weight * (ans.parsed.confidence or 0.0)
                total_weight += weight

                if debug and debug_rows is not None:
                    contrib = weight * avg_score
                    debug_rows.append((prompt, weight, avg_score, contrib))

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

            # 在 verbose / debug 模式下先缓存当前样本的加权计算细节，
            # 由 ReportBuilder 在 Krippendorff α 之后统一打印。
            if debug and debug_rows and debug_info is not None:
                meta = (any_answer.sample_id, any_answer.answer_id, any_answer.run_idx)
                debug_info.append(
                    (
                        meta,
                        debug_rows,
                        final_score,
                        weighted_total,
                        total_weight,
                    )
                )

        return merged_scores, debug_info
