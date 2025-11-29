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

    @property
    def total_score(self) -> float:
        """Aggregate per-rule scores into a single numeric value.

        新版提示词约定每条规则返回 ``score`` 字段（0~5，分数越高表现越好）。
        为了兼容历史缓存数据，这里同时支持两种格式：
        - 如果存在 ``score``，则按浮点数累加；
        - 否则退回到旧版的 ``hit`` 布尔计数（True 记 1 分）。
        """

        total = 0.0
        for check in self.checks:
            if not isinstance(check, dict):
                continue
            if "score" in check:
                try:
                    total += float(check.get("score", 0) or 0)
                except (TypeError, ValueError):
                    # 非法数值直接视作 0 分
                    continue
            elif check.get("hit"):
                total += 1.0
        return total


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
                if cached is not None:
                    raw_response = cached
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
                    if self.cache:
                        self.cache.set(cache_key, raw_response)

                parsed = self._parse_response(raw_response)
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
                    conf = parsed.confidence if parsed else None
                    preview = (answer[:40] + "...") if len(answer) > 40 else answer
                    print(
                        f"[LLMScorer] sample={sample.sample_id} answer={answer_id} "
                        f"run={run_idx + repeat_idx} prompt={prompt_version} "
                        f"temp={temperature} top_k={top_k} top_p={top_p} "
                        f"total_score={total} confidence={conf} | answer_preview={preview}"
                    )
                    print(raw_response)


        return scores

    def _parse_response(self, raw: str) -> Optional[Score]:
        try:
            payload = json.loads(raw)
            checks = payload.get("checks", [])
            confidence = float(payload.get("confidence", 0))
            if isinstance(checks, list):
                return Score(checks=checks, confidence=confidence)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        return None

    def flush_cache(self) -> None:
        if self.cache:
            self.cache.flush()
