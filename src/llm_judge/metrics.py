from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .llm_scorer import AnswerScore


@dataclass
class PairDecision:
    sample_id: str
    run_idx: int
    score_a: float
    score_b: float
    human_winner: str

    @property
    def llm_winner(self) -> Optional[str]:
        if self.score_a == self.score_b:
            return None
        return "A" if self.score_a > self.score_b else "B"


class MetricsComputer:
    """Compute stability, agreement, and alt-test metrics."""

    def krippendorff_alpha_interval(self, scores: Dict[str, List[float]]) -> float:
        """Compute Krippendorff's alpha for interval data.

        ``scores`` maps ``sample_id`` to a list of repeated numeric ratings.
        """

        all_ratings = [rating for values in scores.values() for rating in values if rating is not None]
        if len(all_ratings) <= 1:
            return 0.0

        mean = sum(all_ratings) / len(all_ratings)
        de = sum((rating - mean) ** 2 for rating in all_ratings)
        if de == 0:
            return 1.0

        do = 0.0
        for ratings in scores.values():
            values = [r for r in ratings if r is not None]
            n = len(values)
            if n < 2:
                continue
            local_mean = sum(values) / n
            do += sum((r - local_mean) ** 2 for r in values)
        return 1 - (do / de)

    def build_pair_decisions(self, answers: Iterable[AnswerScore]) -> List[PairDecision]:
        grouped: Dict[Tuple[str, int], Dict[str, AnswerScore]] = defaultdict(dict)
        for ans in answers:
            grouped[(ans.sample_id, ans.run_idx)][ans.answer_id] = ans

        decisions: List[PairDecision] = []
        for (sample_id, run_idx), pair in grouped.items():
            a_score = pair.get("A")
            b_score = pair.get("B")
            if not a_score or not b_score:
                continue
            score_a = a_score.parsed.total_score if a_score.parsed else 0.0
            score_b = b_score.parsed.total_score if b_score.parsed else 0.0
            human_winner = a_score.human_winner or b_score.human_winner
            decisions.append(
                PairDecision(
                    sample_id=sample_id,
                    run_idx=run_idx,
                    score_a=score_a,
                    score_b=score_b,
                    human_winner=human_winner or "",
                )
            )
        return decisions

    def pair_accuracy(self, decisions: Iterable[PairDecision]) -> Tuple[float, float]:
        correct = 0
        total = 0
        ties = 0
        for decision in decisions:
            winner = decision.llm_winner
            if winner is None:
                ties += 1
                continue
            if decision.human_winner and winner == decision.human_winner.upper():
                correct += 1
            total += 1
        acc = correct / total if total else 0.0
        tie_rate = ties / (total + ties) if (total + ties) else 0.0
        return acc, tie_rate

    def alt_test(self, decisions: Iterable[PairDecision]) -> Tuple[float, float]:
        counts_llm = Counter()
        counts_human = Counter()
        for decision in decisions:
            llm_winner = decision.llm_winner
            if llm_winner:
                counts_llm[llm_winner] += 1
            if decision.human_winner:
                counts_human[decision.human_winner.upper()] += 1

        categories = {"A", "B"}
        chi_square = 0.0
        for cat in categories:
            observed = counts_llm.get(cat, 0)
            expected = counts_human.get(cat, 0)
            if expected == 0:
                continue
            chi_square += (observed - expected) ** 2 / expected

        p_value = self._chi_square_sf(chi_square, df=1)
        return chi_square, p_value

    @staticmethod
    def _chi_square_sf(stat: float, df: int) -> float:
        if df != 1:
            # simple fallback using exponential tail approximation
            return math.exp(-0.5 * stat)
        return 1 - math.erf(math.sqrt(stat / 2))
