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
        return sum(1.0 for check in self.checks if check.get("hit"))


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
    ):
        self.caller = caller
        self.prompt_manager = prompt_manager
        self.cache = cache
        self.templater = templater or InputTemplater()

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
                scores.append(
                    AnswerScore(
                        sample_id=sample.sample_id,
                        answer_id=answer_id,
                        human_winner=sample.winner,
                        prompt_version=prompt_version,
                        decode_params={"temperature": temperature, "top_k": top_k, "top_p": top_p},
                        raw_response=raw_response,
                        parsed=self._parse_response(raw_response),
                        run_idx=run_idx + repeat_idx,
                    )
                )
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
