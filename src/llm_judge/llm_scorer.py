from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional

from .cache import CacheKey, JSONCache
from .data_loader import Sample
from .prompt_manager import PromptManager, build_prompt_variables
from .templates import InputTemplater

LLMCaller = Callable[[str, float, int | None, float | None, int], str]

try:  # 在运行环境允许的情况下，从 init_prompt 中导入各 prompt 的维度定义
    from init_prompt import PROMPT_DIM_MAP
except Exception:  # noqa: BLE001
    PROMPT_DIM_MAP: dict[str, List[str]] = {}


@dataclass
class Score:
    checks: List[dict]
    confidence: float

    def _iter_numeric_scores(self) -> List[float]:
        values: List[float] = []
        for check in self.checks:
            if not isinstance(check, dict):
                continue
            if "score" in check:
                try:
                    values.append(float(check.get("score", 0) or 0))
                except (TypeError, ValueError):
                    continue
            elif "hit" in check:
                values.append(1.0 if bool(check.get("hit")) else 0.0)
        return values

    @staticmethod
    def from_dict(payload: Mapping | None) -> "Score | None":
        """从字典结构安全地还原为 ``Score`` 对象。

        主要用于从缓存中读回已经存储好的解析结果。
        """

        if payload is None or not isinstance(payload, Mapping):
            return None

        checks = payload.get("checks", [])
        if not isinstance(checks, list):
            checks = []

        try:
            confidence = float(payload.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            confidence = 0.0

        return Score(checks=checks, confidence=confidence)

    @property
    def total_score(self) -> float:
        """Aggregate per-rule scores into a single numeric value.

        新版提示词约定每条规则返回 ``score`` 字段（0~5，分数越高表现越好）。
        为了兼容历史缓存数据，这里同时支持两种格式：
        - 如果存在 ``score``，则按浮点数累加；
        - 否则退回到旧版的 ``hit`` 布尔计数（True 记 1 分）。
        """

        return sum(self._iter_numeric_scores())

    @property
    def average_score(self) -> float:
        """Average of per-rule numeric scores.

        取每条规则的得分（score 或退回 hit→0/1）的均值，便于不同
        prompt 之间按「平均得分」进行加权融合。
        """

        values = self._iter_numeric_scores()
        return sum(values) / len(values) if values else 0.0


@dataclass
class AnswerScore:
    sample_id: str
    answer_id: str
    human_winner: str
    prompt_version: str
    decode_params: Dict[str, float | int | None]
    raw_response: str
    parsed: Optional[Score]
    run_idx: int


class LLMScorer:
    """Score answers with LLM prompts and parse JSON responses."""

    def __init__(
        self,
        caller: LLMCaller,
        prompt_manager: PromptManager,
        cache: Optional[JSONCache] = None,
        templater: Optional[InputTemplater] = None,
        verbose: bool = False,
        *,
        # 缺失打分维度的处理策略：
        # - "fill_max": 按提示词定义的维度补齐缺失项，默认给满分；
        # - "ignore" : 不补齐，后续指标计算仅基于模型实际输出的维度；
        # - "retry"  : 将缺失视为一次失败，尽量触发重新打分（仅对本次 LLM 调用生效，缓存命中不重试）。
        missing_dims_strategy: str = "fill_max",
        # 在 "fill_max" 策略下，缺失项默认使用的分值（通常为 5.0，即满分）
        missing_full_score: float = 5.0,
    ):
        self.caller = caller
        self.prompt_manager = prompt_manager
        self.cache = cache
        self.templater = templater or InputTemplater()
        # 当 verbose=True 时，在打分阶段输出单条推理的可视化信息
        self.verbose = verbose
        if missing_dims_strategy not in {"fill_max", "ignore", "retry"}:
            raise ValueError(
                "missing_dims_strategy must be one of {'fill_max', 'ignore', 'retry'}; "
                f"got {missing_dims_strategy!r}"
            )
        self.missing_dims_strategy = missing_dims_strategy
        self.missing_full_score = missing_full_score

    # === 维度对齐与缺失项处理相关工具方法 ===

    def _expected_rule_ids_for_prompt(self, prompt_version: str) -> List[str] | None:
        """根据 prompt 版本获取预期的规则项 ID 列表。

        规则来源于 ``init_prompt.PROMPT_DIM_MAP``，这样当提示词发生演化时，
        只需在 init_prompt 中更新一处即可。
        """

        dims = PROMPT_DIM_MAP.get(prompt_version)
        if not dims:
            return None
        # 去重同时保持原有顺序
        seen: set[str] = set()
        ordered: List[str] = []
        for dim in dims:
            if dim not in seen:
                seen.add(dim)
                ordered.append(dim)
        return ordered

    @staticmethod
    def _extract_rule_id(check: Mapping, fallback_index: int) -> str | None:
        """从单条规则项中抽取 rule_id，必要时退回到索引。"""

        rule_id = (
            check.get("rule_id")
            or check.get("name")
            or check.get("rule")
            or None
        )
        if rule_id is None:
            return str(fallback_index)
        return str(rule_id)

    def _ensure_expected_dims(
        self,
        parsed: Optional["Score"],
        expected_rule_ids: Optional[List[str]],
        *,
        prompt_version: str,
        sample_id: str,
        answer_id: str,
        source: str,
    ) -> Optional["Score"]:
        """确保解析结果中覆盖当前 prompt 定义的全部打分维度。

        - 当 ``missing_dims_strategy == "fill_max"`` 时，对缺失维度自动补齐默认满分；
        - 当 ``missing_dims_strategy == "ignore"`` 时，不做改动；
        - 当 ``missing_dims_strategy == "retry"`` 时：
          - 对缓存命中 / 纯解析场景，仅打印提示并退回到 ``fill_max`` 行为；
          - 真正的“触发重试”逻辑在 ``_call_with_retry`` 内完成。
        """

        if parsed is None or not expected_rule_ids:
            return parsed

        present_ids: set[str] = set()
        for idx, check in enumerate(parsed.checks):
            if not isinstance(check, Mapping):
                continue
            rid = self._extract_rule_id(check, idx)
            present_ids.add(rid)

        missing = [rid for rid in expected_rule_ids if rid not in present_ids]
        if not missing:
            return parsed

        # ignore 策略：完全按模型实际输出的维度计算，不补齐
        if self.missing_dims_strategy == "ignore":
            if self.verbose:
                print(
                    "[LLMScorer][INFO] 解析结果存在缺失维度，但按 'ignore' 策略跳过补齐："
                    f"sample={sample_id} answer={answer_id} prompt={prompt_version} "
                    f"missing={missing} source={source}"
                )
            return parsed

        # "retry" 策略在缓存 / 纯解析场景无法重新调用 LLM，这里退回到 fill_max 行为，
        # 避免因为历史缓存或离线处理而中断流水线。
        if self.missing_dims_strategy == "retry" and self.verbose:
            print(
                "[LLMScorer][INFO] 解析结果存在缺失维度，但当前来源不支持重试，"
                "退回到 'fill_max' 补齐策略："
                f"sample={sample_id} answer={answer_id} prompt={prompt_version} "
                f"missing={missing} source={source}"
            )

        # "fill_max" 策略：为所有缺失规则项补齐一条默认满分记录
        new_checks: List[dict] = []
        for idx, check in enumerate(parsed.checks):
            if isinstance(check, Mapping):
                # 拷贝一份，避免意外修改外部引用
                new_checks.append(dict(check))
            else:
                new_checks.append({"value": check})

        for rid in missing:
            auto_check = {
                "rule_id": rid,
                "score": self.missing_full_score,
                "reason": "模型未显式输出该维度，按配置使用默认满分补齐。",
                "excerpt": "",
                "auto_filled": True,
            }
            new_checks.append(auto_check)

        if self.verbose:
            print(
                "[LLMScorer][INFO] 为缺失维度补齐默认满分："
                f"sample={sample_id} answer={answer_id} prompt={prompt_version} "
                f"missing={missing} source={source}"
            )

        return Score(checks=new_checks, confidence=parsed.confidence)

    def score_sample(
        self,
        sample: Sample,
        prompt_version: str,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        run_idx: int,
        repeats: int = 1,
    ) -> List[AnswerScore]:
        tpl_result = self.templater(sample)
        scores: List[AnswerScore] = []
        expected_rule_ids = self._expected_rule_ids_for_prompt(prompt_version)
        for answer_id, answer in ("A", sample.a_answer), ("B", sample.b_answer):
            for repeat_idx in range(repeats):
                raw_response: str
                prompt = self.prompt_manager.render(
                    prompt_version,
                    build_prompt_variables(
                        input_data=tpl_result.input_data,
                        modules_block=tpl_result.modules_block,
                        answer=answer,
                    ),
                )
                # 为了提高鲁棒性，这里对大模型调用和 JSON 解析增加最多 10 次的重试机制。
                # 在网络波动、服务端 5xx 或模型返回非 JSON 等情况下，不至于直接中断整条评估流水线。
                raw_response, parsed = self._call_with_retry(
                    prompt=prompt,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=run_idx + repeat_idx,
                    sample_id=sample.sample_id,
                    answer_id=answer_id,
                    prompt_version=prompt_version,
                    expected_rule_ids=expected_rule_ids,
                )
                if self.cache:
                    cache_key = CacheKey(
                        prompt_version=prompt_version,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        query_id=sample.sample_id,
                        answer_id=answer_id,
                        run_idx=run_idx + repeat_idx,
                    )
                    self.cache.set(
                        cache_key,
                        self._build_cache_entry(
                            sample=sample,
                            answer=answer,
                            answer_id=answer_id,
                            prompt_version=prompt_version,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            run_idx=run_idx + repeat_idx,
                            tpl_result=tpl_result,
                            raw_response=raw_response,
                            parsed=parsed,
                        ),
                    )

                score_obj = AnswerScore(
                    sample_id=sample.sample_id,
                    answer_id=answer_id,
                    human_winner=sample.winner,
                    prompt_version=prompt_version,
                    decode_params={"temperature": temperature, "top_k": top_k, "top_p": top_p},
                    raw_response=raw_response,
                    parsed=parsed,
                    run_idx=run_idx + repeat_idx,
                )
                scores.append(score_obj)

                if self.verbose:
                    total = parsed.total_score if parsed else None
                    average = parsed.average_score if parsed else None
                    conf = parsed.confidence if parsed else None
                    seed = run_idx
                    repeat_no = repeat_idx + 1
                    preview = (answer[:40] + "...") if len(answer) > 40 else answer
                    print(
                        f"[LLMScorer] sample={sample.sample_id} answer={answer_id} "
                        f"seed={seed} repeat={repeat_no}/{repeats} "
                        f"prompt={prompt_version} temp={temperature} top_k={top_k} top_p={top_p} "
                        f"average_score={average:.2f} total_score={total:.2f} confidence={conf:.2f} | "
                        f"answer_preview={preview}"
                    )
                    # 使用 ANSI 转义序列将 parsed 的详细内容打印为灰色，便于与主日志区分。
                    print(f"\033[90m{parsed}\033[0m")

        return scores

    def _build_cache_entry(
        self,
        sample: Sample,
        answer: str,
        answer_id: str,
        prompt_version: str,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        run_idx: int,
        tpl_result,
        raw_response: str,
        parsed: Optional[Score],
    ) -> Dict:
        """构造写入缓存文件的完整记录。

        目标：在 ``--cache`` 指定的 JSON 文件中，详细记录每次“大模型打分调用”的
        - 原始回复内容
        - 解析后的评分结果
        - 相关的实验 / 配置信息
        - 输入样本与候选答案信息
        """

        parsed_payload: Optional[Dict] = None
        if parsed is not None:
            parsed_payload = {
                "checks": parsed.checks,
                "confidence": parsed.confidence,
                "total_score": parsed.total_score,
                "average_score": parsed.average_score,
            }

        return {
            "raw_response": raw_response,
            "parsed": parsed_payload,
            "meta": {
                "sample_id": sample.sample_id,
                "answer_id": answer_id,
                "answer": answer,
                "human_winner": sample.winner,
                "prompt_version": prompt_version,
                "decode_params": {
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                },
                # run_idx 体现了 EvaluationPipeline 中的 seed / 轮次信息
                "run_idx": run_idx,
                # 保留原始 CSV 中的额外字段，便于线下分析
                "sample_extra": sample.extra,
            },
            "input": {
                "query": sample.query,
                "last_answer_phone": sample.last_answer_phone,
                "modules_block": sample.modules_block,
                # 模板展开后的结构化输入（如果后续模板逻辑扩展，这里也能完整记录）
                "templated_input_data": getattr(tpl_result, "input_data", None),
                "templated_modules_block": getattr(tpl_result, "modules_block", None),
            },
        }

    def _parse_response(self, raw: str) -> Optional[Score]:
        try:
            import json_repair
            payload = json_repair.loads(raw)
            checks = payload.get("checks", [])
            confidence = float(payload.get("confidence", 0))
            if isinstance(checks, list):
                return Score(checks=checks, confidence=confidence)
        except (json.JSONDecodeError, TypeError, ValueError):
            raise ValueError(f"Failed to parse JSON response: {raw}")
        return None

    def _call_with_retry(
        self,
        prompt: str,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        seed: int,
        *,
        sample_id: str,
        answer_id: str,
        prompt_version: str,
        max_retries: int = 10,
        expected_rule_ids: Optional[List[str]] = None,
    ) -> tuple[str, Optional[Score]]:
        """带重试的大模型调用封装。

        - 最大重试 ``max_retries`` 次（包含第一次尝试）；
        - 每次失败都会打印一条告警信息，但不中断整条评估流程；
        - 如果始终失败，则返回 ``("", None)``，后续指标计算阶段会将其视为“无评分”样本。
        """

        last_error: Exception | None = None
        last_raw: str = ""

        for attempt in range(1, max_retries + 1):
            try:
                raw = self.caller(prompt, temperature, top_k, top_p, seed)
                last_raw = str(raw)
                parsed = self._parse_response(last_raw)

                # 如果模型成功返回，但遗漏了部分维度，在 "retry" 策略下可以视作一次失败，
                # 通过抛出异常进入重试分支；否则在本轮内直接按配置补齐/忽略。
                if expected_rule_ids:
                    present_ids: set[str] = set()
                    if parsed is not None:
                        for idx, check in enumerate(parsed.checks):
                            if not isinstance(check, Mapping):
                                continue
                            rid = self._extract_rule_id(check, idx)
                            present_ids.add(rid)
                    missing = [rid for rid in expected_rule_ids if rid not in present_ids]
                    if missing and self.missing_dims_strategy == "retry" and attempt < max_retries:
                        if self.verbose:
                            print(
                                "[LLMScorer][WARN] 模型返回中缺失部分维度，按照 'retry' 策略重试："
                                f"sample={sample_id} answer={answer_id} prompt={prompt_version} "
                                f"attempt={attempt}/{max_retries} missing={missing}"
                            )
                        raise ValueError(f"Missing dimensions in model response: {missing}")

                    # 在非 retry 或最后一次重试时，按普通逻辑对缺失维度进行处理
                    parsed = self._ensure_expected_dims(
                        parsed,
                        expected_rule_ids,
                        prompt_version=prompt_version,
                        sample_id=sample_id,
                        answer_id=answer_id,
                        source="llm",
                    )

                return last_raw, parsed
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(
                    "[LLMScorer][WARN] 调用大模型失败："
                    f"sample={sample_id} answer={answer_id} prompt={prompt_version} "
                    f"attempt={attempt}/{max_retries} error={exc!r}"
                )

        # 所有重试均失败，打印最终错误并返回空结果，避免中断整体流程。
        if last_error is not None:
            print(
                "[LLMScorer][WARN] 调用大模型连续失败已达上限，将跳过本次打分："
                f"sample={sample_id} answer={answer_id} prompt={prompt_version} "
                f"max_retries={max_retries} last_error={last_error!r}"
            )
        return last_raw, None

    def flush_cache(self) -> None:
        if self.cache:
            self.cache.flush()
