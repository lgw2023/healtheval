from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional

from .cache import CacheKey, JSONCache
from .data_loader import Sample
from .prompt_manager import PromptManager, build_prompt_variables
from .templates import InputTemplater

LLMCaller = Callable[[str, float, int | None, float | None, int], str]


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
    ):
        self.caller = caller
        self.prompt_manager = prompt_manager
        self.cache = cache
        self.templater = templater or InputTemplater()
        # 当 verbose=True 时，在打分阶段输出单条推理的可视化信息
        self.verbose = verbose

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
        for answer_id, answer in ("A", sample.a_answer), ("B", sample.b_answer):
            for repeat_idx in range(repeats):
                cache_key = CacheKey(
                    prompt_version=prompt_version,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    query_id=sample.sample_id,
                    answer_id=answer_id,
                    run_idx=run_idx + repeat_idx,
                )
                cached = self.cache.get(cache_key) if self.cache else None
                raw_response: str

                # 1. 命中缓存：支持两种历史格式
                if cached is not None:
                    if isinstance(cached, dict):
                        # 新版：结构化缓存
                        raw_response = str(cached.get("raw_response", ""))
                        # 优先使用结构化 parsed 字段，还原为 Score；否则退回到重新解析 raw_response
                        parsed = Score.from_dict(cached.get("parsed")) or self._parse_response(raw_response)
                    else:
                        # 旧版：仅字符串形式的 LLM 原始回复
                        raw_response = str(cached)
                        parsed = self._parse_response(raw_response)
                        # 兼容迁移：在读取旧缓存时，自动升级为结构化格式，写回 cache 文件
                        if self.cache:
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

                # 2. 未命中缓存：实际调用大模型并写入结构化缓存
                else:
                    prompt = self.prompt_manager.render(
                        prompt_version,
                        build_prompt_variables(
                            input_data=tpl_result.input_data,
                            modules_block=tpl_result.modules_block,
                            answer=answer,
                        ),
                    )
                    raw_response = self.caller(prompt, temperature, top_k, top_p, run_idx + repeat_idx)
                    parsed = self._parse_response(raw_response)
                    if self.cache:
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
                        f"average_score={average} total_score={total} confidence={conf} | "
                        f"answer_preview={preview}"
                    )
                    print(parsed)


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

    def flush_cache(self) -> None:
        if self.cache:
            self.cache.flush()
