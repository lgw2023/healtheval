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
    # 额外统计信息，便于在命令行中做可视化输出
    num_answers: int
    num_decisions: int
    llm_winner_counts: dict
    human_winner_counts: dict


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

        llm_counts = {"A": 0, "B": 0}
        human_counts = {"A": 0, "B": 0}
        for d in decisions:
            if d.llm_winner in llm_counts:
                llm_counts[d.llm_winner] += 1
            hw = d.human_winner.upper() if d.human_winner else ""
            if hw in human_counts:
                human_counts[hw] += 1

        return EvaluationReport(
            alpha=alpha,
            pair_accuracy=pair_acc,
            tie_rate=tie_rate,
            chi_square=chi_square,
            alt_test_p=p_value,
            num_answers=len(answers_list),
            num_decisions=len(decisions),
            llm_winner_counts=llm_counts,
            human_winner_counts=human_counts,
        )

    @staticmethod
    def _aggregate_scores(answers: List[AnswerScore]) -> Dict[str, List[float]]:
        stability: Dict[str, List[float]] = {}
        for answer in answers:
            score = answer.parsed.total_score if answer.parsed else None
            stability.setdefault(answer.sample_id + answer.answer_id, []).append(score)
        return stability

    @staticmethod
    def to_pretty_text(report: EvaluationReport) -> str:
        """将指标结果格式化为更易读的文本，可用于命令行可视化输出。"""

        lines = [
            f"alpha={report.alpha:.4f}",
            f"pair_acc={report.pair_accuracy:.4f}",
            f"tie_rate={report.tie_rate:.4f}",
            f"chi_square={report.chi_square:.4f}",
            f"alt_test_p={report.alt_test_p:.4f}",
            f"samples_decisions={report.num_decisions} (answers={report.num_answers})",
            "human_winner_dist: "
            f"A={report.human_winner_counts.get('A', 0)}, "
            f"B={report.human_winner_counts.get('B', 0)}",
            "llm_winner_dist:   "
            f"A={report.llm_winner_counts.get('A', 0)}, "
            f"B={report.llm_winner_counts.get('B', 0)}",
        ]
        return "\n".join(lines)
