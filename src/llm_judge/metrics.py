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
    """Compute stability, agreement, and alt-test metrics.

    为了**可视化 task.md 中描述的三类指标计算逻辑**，本类在 debug 模式下会打印
    详细的中间结果，包括：
    - 单条打分稳定性：每个样本在多次采样下的打分列表、DO/DE 分解和最终 α；
    - 成对比较一致性：每条样本 (A,B) 的平均得分、LLM 判定胜者、人工 winner 及是否命中；
    - Alt-Test：人工与 LLM 在 A/B 胜负上的计数、卡方每一项贡献及最终 p-value。
    """

    def __init__(self, debug: bool = False) -> None:
        # 当 debug=True 时，会在各个指标计算函数中输出事无巨细的中间计算过程
        self.debug = debug

    def krippendorff_alpha_interval(self, scores: Dict[str, List[float]]) -> float:
        """Compute Krippendorff's alpha for interval data.

        ``scores`` 将「评分单元 ID」映射到该单元在多次采样下的数值型得分列表。

        在当前项目中，「评分单元」通常可以理解为：
        - 早期实现：一个 (sample_id, answer_id) 在多次打分下的 total_score；
        - 现行实现：一个 (sample_id, answer_id, rule_id) 对应的一条规则项得分，
          即「单条打分中的某一项键值对」在多次采样下的得分。

        对应 task.md 的「单条打分稳定性：Krippendorff’s α」。
        """

        if self.debug:
            print("\n[Metrics] ==== 单条打分稳定性：Krippendorff’s α（区间型）====")
            print(f"[Metrics] 评分单元数量（如 query+answer+rule 组合）={len(scores)}")
            for unit_id, ratings in scores.items():
                print(f"[Metrics]  单位 {unit_id!r} 的多次打分列表: {ratings}")

        all_ratings = [rating for values in scores.values() for rating in values if rating is not None]
        if len(all_ratings) <= 1:
            if self.debug:
                print("[Metrics] 可用打分总数 <= 1，α 直接记为 0.0")
            return 0.0

        mean = sum(all_ratings) / len(all_ratings)
        de = sum((rating - mean) ** 2 for rating in all_ratings)
        if self.debug:
            print(f"[Metrics] 所有打分的全局均值 mean={mean:.6f}")
            print(f"[Metrics] 期望误差平方和 DE={de:.6f}")

        if de == 0:
            if self.debug:
                print("[Metrics] 全体打分完全一致（DE=0），α=1.0")
            return 1.0

        do = 0.0
        for unit_id, ratings in scores.items():
            values = [r for r in ratings if r is not None]
            n = len(values)
            if n < 2:
                if self.debug:
                    print(f"[Metrics]  单位 {unit_id!r} 有效打分数 < 2，跳过 DO 贡献")
                continue
            local_mean = sum(values) / n
            local_do = sum((r - local_mean) ** 2 for r in values)
            do += local_do
            if self.debug:
                print(
                    f"[Metrics]  单位 {unit_id!r}: 局部均值={local_mean:.6f}, "
                    f"局部误差平方和 DO_unit={local_do:.6f}"
                )

        alpha = 1 - (do / de)
        if self.debug:
            print(f"[Metrics] 总体观测误差平方和 DO={do:.6f}")
            print(f"[Metrics] Krippendorff’s α = 1 - DO/DE = {alpha:.6f}")
        return alpha

    def build_pair_decisions(self, answers: Iterable[AnswerScore]) -> List[PairDecision]:
        """将每条样本在一次运行中的 (A,B) 平均得分配成成对决策。

        对应 task.md 的「成对比较一致性：与 winner 的方向是否一致」。
        """

        grouped: Dict[Tuple[str, int], Dict[str, AnswerScore]] = defaultdict(dict)
        for ans in answers:
            grouped[(ans.sample_id, ans.run_idx)][ans.answer_id] = ans

        decisions: List[PairDecision] = []
        for (sample_id, run_idx), pair in grouped.items():
            a_score = pair.get("A")
            b_score = pair.get("B")
            if not a_score or not b_score:
                continue
            score_a = a_score.parsed.average_score if a_score.parsed else 0.0
            score_b = b_score.parsed.average_score if b_score.parsed else 0.0
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

        if self.debug:
            print("\n[Metrics] ==== 成对比较：每条样本 (A,B) 的平均得分与胜负方向 ====")
            for d in decisions:
                print(
                    "[Metrics]  "
                    f"sample={d.sample_id}, run={d.run_idx}, "
                    f"avg_score_A={d.score_a:.4f}, avg_score_B={d.score_b:.4f}, "
                    f"llm_winner={d.llm_winner}, human_winner={d.human_winner}"
                )
        return decisions

    def pair_accuracy(self, decisions: Iterable[PairDecision]) -> Tuple[float, float]:
        """计算成对一致性准确率以及平局率。"""

        correct = 0
        total = 0
        ties = 0

        if self.debug:
            print("\n[Metrics] ==== 成对比较一致性：pair-accuracy 与平局率 ====")

        for decision in decisions:
            winner = decision.llm_winner
            if winner is None:
                ties += 1
                if self.debug:
                    print(
                        "[Metrics]  平局样本 "
                        f"(sample={decision.sample_id}, run={decision.run_idx}): "
                        f"score_A={decision.score_a:.4f}, score_B={decision.score_b:.4f}"
                    )
                continue

            is_correct = decision.human_winner and winner == decision.human_winner.upper()
            if is_correct:
                correct += 1
            total += 1

            if self.debug:
                print(
                    "[Metrics]  样本 "
                    f"(sample={decision.sample_id}, run={decision.run_idx}): "
                    f"llm_winner={winner}, human_winner={decision.human_winner}, "
                    f"命中={'YES' if is_correct else 'NO'}"
                )

        acc = correct / total if total else 0.0
        tie_rate = ties / (total + ties) if (total + ties) else 0.0

        if self.debug:
            print(
                "[Metrics]  统计汇总: "
                f"有明确胜负的样本数={total}, 平局数={ties}, "
                f"pair-accuracy={acc:.6f}, tie_rate={tie_rate:.6f}"
            )
        return acc, tie_rate

    def alt_test(self, decisions: Iterable[PairDecision]) -> Tuple[float, float]:
        """Alt-Test：LLM 胜负分布与人工胜负分布是否存在显著差异。

        使用一个简化的 2 类卡方检验：
        - 统计 human 与 llm 在 {A,B} 上的胜场；
        - 以 human 的分布作为期望，llm 的分布作为观测；
        - 输出 (chi_square, p_value)。
        """

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

        if self.debug:
            print("\n[Metrics] ==== Alt-Test：总体数据一致性（卡方近似）====")
            print(f"[Metrics] 人工胜负计数 human_counts={dict(counts_human)}")
            print(f"[Metrics] LLM 胜负计数   llm_counts={dict(counts_llm)}")

        for cat in categories:
            observed = counts_llm.get(cat, 0)
            expected = counts_human.get(cat, 0)
            if expected == 0:
                if self.debug:
                    print(
                        f"[Metrics]  类别 {cat} 在人工标注中计数为 0，"
                        "跳过该类别的卡方贡献（避免除以 0）。"
                    )
                continue
            contrib = (observed - expected) ** 2 / expected
            chi_square += contrib
            if self.debug:
                print(
                    f"[Metrics]  类别 {cat}: observed={observed}, expected={expected}, "
                    f"贡献 chi^2={contrib:.6f}"
                )

        p_value = self._chi_square_sf(chi_square, df=1)

        if self.debug:
            print(f"[Metrics] 卡方统计量 chi_square={chi_square:.6f}")
            print(f"[Metrics] Alt-Test 近似 p-value={p_value:.6f}")
        return chi_square, p_value

    @staticmethod
    def _chi_square_sf(stat: float, df: int) -> float:
        if df != 1:
            # simple fallback using exponential tail approximation
            return math.exp(-0.5 * stat)
        return 1 - math.erf(math.sqrt(stat / 2))
