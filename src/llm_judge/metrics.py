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
    human_winner: str  # 归一化后仅取 {"A", "B", "SAME", ""} 这几类值

    @property
    def llm_winner(self) -> str:
        """LLM 的胜负方向，取值限定为 {"A", "B", "SAME"}。

        - "A" / "B": LLM 判定其中一方得分更高；
        - "SAME"   : LLM 判定 A/B 打分完全相同（平局）。

        这样在后续计算和日志中，可以与人工标注 winner 的三种取值
        {"A", "B", "SAME"} 对齐，避免平局样本被当作 None 特殊分支处理。
        """

        # 当 A/B 平均得分非常接近时，可以视为“无明显胜负方向”，记作平局 SAME。
        # 这里采用一个对称的阈值：若 |score_a - score_b| < 0.1，则认为模型判为 SAME。
        diff = self.score_a - self.score_b
        if abs(diff) < 0.1:
            return "SAME"
        return "A" if diff > 0 else "B"


def _normalize_human_winner(label: str | None) -> str:
    """将 CSV / 缓存中的 winner 归一化到 {\"A\", \"B\", \"SAME\", \"\"}.

    - 支持大小写无关的 \"a\" / \"b\"；
    - 对于打分相同的情况，约定使用 \"same\" / \"SAME\"，统一映射为 \"SAME\"；
    - 其它非法值统一记为空字符串 \"\"，在后续计算中会被自动跳过。
    """

    if not label:
        return ""
    value = str(label).strip()
    if not value:
        return ""

    upper = value.upper()
    if upper in {"A", "B"}:
        return upper
    if upper in {"SAME", "TIE"}:
        return "SAME"
    return ""


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

            # 如果 unit_id 形如 "sample_id+answer_id:rule_id"
            # （例如 "0A:PERSONAL_DATA_MISMATCH"），则按照：
            #   - 行：sample+answer（如 0A / 0B）
            #   - 列：各规则项（rule_id）
            #   - 单元格：该样本、该回答、该规则在多次采样下的评分列表
            # 的形式输出一个稀疏矩阵，以减少日志行数、便于人眼横向对比。
            has_structured_unit_id = any(":" in str(uid) for uid in scores.keys())

            if has_structured_unit_id:
                # 先聚合为 matrix[row_key][rule_id] = ratings
                matrix: Dict[str, Dict[str, List[float]]] = {}
                all_rules: set[str] = set()
                for unit_id, ratings in scores.items():
                    uid_str = str(unit_id)
                    row_key, rule_id = uid_str.split(":", 1)
                    matrix.setdefault(row_key, {})[rule_id] = ratings
                    all_rules.add(rule_id)

                row_keys = sorted(matrix.keys())
                rule_ids = sorted(all_rules)

                print(
                    "[Metrics]  按样本+回答 (如 0A/0B) 作为行，规则项作为列，"
                    "每个单元格为 repeats 得到的分值列表"
                )
                # 表头
                header = ["sample_answer"] + rule_ids
                print("[Metrics]  " + " | ".join(header))
                print("[Metrics]  " + "-" * 110)

                # 每一行对应一个样本+回答（例如 0A, 0B）
                for row_key in row_keys:
                    row_values = [row_key]
                    rule_map = matrix.get(row_key, {})
                    for rule_id in rule_ids:
                        ratings = rule_map.get(rule_id)
                        row_values.append(str(ratings) if ratings is not None else "-")
                    print("[Metrics]  " + " | ".join(row_values))
            else:
                # 兜底：如果 unit_id 不是结构化的 "sample:rule" 形式，退回到旧版逐行输出
                print("[Metrics]  各评分单元的打分列表（表格形式）")
                print("[Metrics]  " + "-" * 90)
                print(
                    "[Metrics]  "
                    f"{'unit_id':<40} | ratings"
                )
                print("[Metrics]  " + "-" * 90)
                for unit_id, ratings in scores.items():
                    print(
                        "[Metrics]  "
                        f"{str(unit_id):<40} | {ratings}"
                    )

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
        # 为了表格化输出，先把每个单元的 DO 相关中间结果缓存下来
        debug_rows = [] if self.debug else None
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
                debug_rows.append(
                    (str(unit_id), n, local_mean, local_do)
                )

        # 将 DO 分解的明细以表格形式统一打印
        if self.debug and debug_rows:
            # 复用与上方打分列表相同的“样本+回答 / 规则项”矩阵逻辑，减少日志行数
            has_structured_unit_id = any(":" in row[0] for row in debug_rows)

            if has_structured_unit_id:
                # matrix[row_key][rule_id] = (n_valid, local_mean, DO_unit)
                matrix: Dict[str, Dict[str, Tuple[int, float, float]]] = {}
                all_rules: set[str] = set()
                for unit_id, n, local_mean, local_do in debug_rows:
                    row_key, rule_id = unit_id.split(":", 1)
                    matrix.setdefault(row_key, {})[rule_id] = (n, local_mean, local_do)
                    all_rules.add(rule_id)

                row_keys = sorted(matrix.keys())
                rule_ids = sorted(all_rules)

                print("[Metrics]  各评分单元 DO 分解明细（按样本+回答 / 规则矩阵展示）")
                print(
                    "[Metrics]  单元格内容为 "
                    "n_valid/mean/DO_unit，例：2/4.5000/0.5000"
                )
                # 表头
                header = ["sample_answer"] + rule_ids
                print("[Metrics]  " + " | ".join(header))
                print("[Metrics]  " + "-" * 110)

                for row_key in row_keys:
                    row_values = [row_key]
                    rule_map = matrix.get(row_key, {})
                    for rule_id in rule_ids:
                        triplet = rule_map.get(rule_id)
                        if triplet is None:
                            cell = "-"
                        else:
                            n, local_mean, local_do = triplet
                            cell = f"{n}/{local_mean:.4f}/{local_do:.4f}"
                        row_values.append(cell)
                    print("[Metrics]  " + " | ".join(row_values))
            else:
                # 兜底：保持旧的逐 unit_id 展开展示
                print("[Metrics]  各评分单元 DO 分解明细（表格形式）")
                print("[Metrics]  " + "-" * 110)
                print(
                    "[Metrics]  "
                    f"{'unit_id':<40} | {'n_valid':>7} | {'local_mean':>12} | {'DO_unit':>12}"
                )
                print("[Metrics]  " + "-" * 110)
                for unit_id, n, local_mean, local_do in debug_rows:
                    print(
                        "[Metrics]  "
                        f"{unit_id:<40} | {n:>7d} | {local_mean:>12.6f} | {local_do:>12.6f}"
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
            # 兼容 CSV / 缓存中的大小写与额外取值（如 \"same\"）：
            # - \"A\" / \"B\" → 方向性胜负；
            # - \"SAME\"      → 人工认为 A/B 打分相同（无方向性胜负）；
            # - 其它值       → 视为缺失，后续指标中会被自动跳过。
            raw_human_winner = a_score.human_winner or b_score.human_winner
            human_winner = _normalize_human_winner(raw_human_winner)
            decisions.append(
                PairDecision(
                    sample_id=sample_id,
                    run_idx=run_idx,
                    score_a=score_a,
                    score_b=score_b,
                    human_winner=human_winner,
                )
            )

        if self.debug:
            print("\n[Metrics] ==== 成对比较：每条样本 (A,B) 的平均得分与胜负方向 ====")
            for d in decisions:
                print(
                    "[Metrics]  "
                    f"sample={d.sample_id}, seed={d.run_idx}, "
                    f"weighted_avg_score_A={d.score_a:.4f}, weighted_avg_score_B={d.score_b:.4f}, "
                    f"llm_winner={d.llm_winner}, human_winner={d.human_winner}"
                )
        return decisions

    def pair_accuracy(self, decisions: Iterable[PairDecision]) -> Tuple[float, float]:
        """计算成对一致性准确率以及平局率。

        - pair-accuracy：将人工 winner ∈ {"A","B","SAME"} 视作三分类标签，
          LLM winner 也统一为 {"A","B","SAME"}，直接按「预测 == 标注」计算准确率；
        - tie_rate：在所有纳入统计的样本中，LLM 判定为 "SAME"（平局）的占比。

        非法或缺失标注（如空字符串）在计算中会被自动跳过。
        """

        correct = 0  # 预测与人工完全一致（含 SAME）的样本数
        total = 0    # 纳入统计的样本总数（人工 winner 合法）
        ties = 0     # 其中 LLM 判定为 "SAME" 的样本数

        if self.debug:
            print("\n[Metrics] ==== 成对比较一致性：pair-accuracy 与平局率 ====")

        for decision in decisions:
            winner = decision.llm_winner
            human = decision.human_winner

            # 仅当人工 winner 为合法取值 {"A","B","SAME"} 时才纳入统计；
            # 其它情况（如空字符串、未知取值）统一跳过。
            if human not in {"A", "B", "SAME"}:
                if self.debug:
                    print(
                        "[Metrics]  跳过 winner 标注异常的样本 "
                        f"(sample={decision.sample_id}, seed={decision.run_idx}): "
                        f"human_winner={human!r}"
                    )
                continue

            total += 1
            is_tie = winner == "SAME"
            if is_tie:
                ties += 1

            is_correct = winner == human
            if is_correct:
                correct += 1

            if self.debug:
                print(
                    "[Metrics]  样本 "
                    f"(sample={decision.sample_id}, seed={decision.run_idx}): "
                    f"llm_winner={winner}, human_winner={human}, "
                    f"命中={'YES' if is_correct else 'NO'}"
                )

        acc = correct / total if total else 0.0
        tie_rate = ties / total if total else 0.0

        if self.debug:
            print(
                "[Metrics]  统计汇总: "
                f"纳入统计的样本数={total}, 其中平局数={ties}, "
                f"pair-accuracy={acc:.6f}, tie_rate={tie_rate:.6f}"
            )
        return acc, tie_rate

    def alt_test(self, decisions: Iterable[PairDecision]) -> Tuple[float, float]:
        """Alt-Test：LLM 胜负分布与人工胜负分布是否存在显著差异。

        使用一个简化的卡方检验：
        - 统计 human 与 llm 在 {A,B,SAME} 上的计数（其中 SAME 代表平局样本）；
        - 以 human 的分布作为期望，llm 的分布作为观测；
        - 输出 (chi_square, p_value)。
        """

        counts_llm = Counter()
        counts_human = Counter()
        for decision in decisions:
            llm_winner = decision.llm_winner
            human = decision.human_winner

            # Alt-Test 同时考虑方向性胜负 {A, B} 以及平局 SAME 的总体分布：
            # - LLM 统计 llm_winner ∈ {A, B, SAME}；
            # - 人工统计 human_winner ∈ {A, B, SAME}。
            if llm_winner in {"A", "B", "SAME"}:
                counts_llm[llm_winner] += 1
            if human in {"A", "B", "SAME"}:
                counts_human[human] += 1

        categories = ["A", "B", "SAME"]
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

        # 自由度 df = 有效类别数 - 1（仅统计在人工标注中出现过的类别）
        effective_cats = [c for c in categories if counts_human.get(c, 0) > 0]
        df = max(len(effective_cats) - 1, 1)
        p_value = self._chi_square_sf(chi_square, df=df)

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
