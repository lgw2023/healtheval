from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .llm_scorer import AnswerScore
from .metrics import MetricsComputer, PairDecision


@dataclass
class EvaluationReport:
    alpha: float
    pair_accuracy: float
    tie_rate: float
    chi_square: float
    alt_test_p: float


class ReportBuilder:
    def __init__(self, metrics: MetricsComputer | None = None):
        self.metrics = metrics or MetricsComputer()

    def summarize(self, answers: Iterable[AnswerScore]) -> EvaluationReport:
        answers_list = list(answers)
        stability = self._aggregate_scores(answers_list)
        alpha = self.metrics.krippendorff_alpha_interval(stability)
        decisions: List[PairDecision] = self.metrics.build_pair_decisions(answers_list)
        pair_acc, tie_rate = self.metrics.pair_accuracy(decisions)
        chi_square, p_value = self.metrics.alt_test(decisions)
        return EvaluationReport(
            alpha=alpha,
            pair_accuracy=pair_acc,
            tie_rate=tie_rate,
            chi_square=chi_square,
            alt_test_p=p_value,
        )

    @staticmethod
    def _aggregate_scores(answers: List[AnswerScore]) -> Dict[str, List[float]]:
        stability: Dict[str, List[float]] = {}
        for answer in answers:
            score = answer.parsed.total_score if answer.parsed else None
            stability.setdefault(answer.sample_id + answer.answer_id, []).append(score)
        return stability
