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
    def __init__(self, metrics: MetricsComputer | None = None, debug_metrics: bool = False):
        """构建评估报告的辅助类。

        - ``metrics``：可注入自定义的 ``MetricsComputer`` 实现；
        - ``debug_metrics``：当为 True 时，将在指标计算过程中输出与 task.md 对应的
          详细中间计算步骤，便于可视化理解。
        """

        # 如果外部没有显式传入 MetricsComputer，则根据 debug_metrics 创建默认实例
        self.metrics = metrics or MetricsComputer(debug=debug_metrics)
        self._debug = debug_metrics
    def summarize(
        self,
        answers: Iterable[AnswerScore],
        answers_for_alpha: Iterable[AnswerScore] | None = None,
        weight_debug: list | None = None,
    ) -> EvaluationReport:
        """根据一组打分结果汇总评估指标。

        参数含义：
        - ``answers``：用于成对比较、Alt-Test 等「最终决策」相关指标的答案集合，
          在当前项目中通常是各个 prompt 按权重融合后的加权结果；
        - ``answers_for_alpha``：用于「单条打分稳定性 (Krippendorff’s α)」的
          原始评分集合。如果为 None，则退回使用 ``answers`` 本身。

        这样可以实现：**α 基于原始 GROUND/STRUCT prompt 的逐规则打分计算，
        而成对比较和 Alt-Test 基于加权后的最终得分计算**，两者解耦。
        """

        answers_list = list(answers)
        alpha_source_list = list(answers_for_alpha) if answers_for_alpha is not None else answers_list
        if self._debug:
            print("\n[Report] ===== 开始计算当前配置的三类指标 =====")
            print(f"[Report] 样本答案条数（含 A/B 和多轮采样）={len(answers_list)}")
            print("[Report] 对应 task.md 中的三个部分：")
            print("          1) 单条打分稳定性 (Krippendorff’s α)")
            print("          2) 成对比较一致性 (pair-accuracy + tie 率)")
            print("          3) 总体数据一致性 (Alt-Test, 卡方统计与 p-value)")

        # α：基于更细粒度的原始评分单元（通常是各 prompt、各规则项的得分）
        stability = self._aggregate_scores(alpha_source_list)
        alpha = self.metrics.krippendorff_alpha_interval(stability)

        # 在 Krippendorff’s α 之后统一打印「加权融合：原始 prompt 得分与权重贡献」
        # 的中间可视化结果，便于与后续 pair-accuracy / Alt-Test 的指标串联阅读。
        if self._debug and weight_debug:
            for meta, debug_rows, final_score, weighted_total, total_weight in weight_debug:
                sample_id, answer_id, run_idx = meta
                print("\n[Metrics] ==== 加权融合：原始 prompt 得分与权重贡献 ====")
                print(
                    "[Metrics]  加权样本: "
                    f"sample={sample_id}, answer={answer_id}, seed={run_idx}"
                )
                for prompt, w, avg, contrib in debug_rows:
                    print(
                        "[Metrics]   - "
                        f"prompt={prompt}, weight={w:.4f}, "
                        f"avg_score_before_weight={avg:.4f}, "
                        f"weight * avg_score={contrib:.4f}"
                    )
                print(
                    "[Metrics]   => final_weighted_score="
                    f"{final_score:.4f}  "
                    "(sum(weight * avg_score) / sum(weight) = "
                    f"{weighted_total:.4f} / {total_weight:.4f})"
                )

        # 成对比较与 Alt-Test：仍然基于「最终决策」所用的答案集合
        decisions: List[PairDecision] = self.metrics.build_pair_decisions(answers_list)
        pair_acc, tie_rate = self.metrics.pair_accuracy(decisions)
        chi_square, p_value = self.metrics.alt_test(decisions)

        # 统计 LLM 在 {A, B, SAME} 上的判定次数，用于总体 winner 分布展示；
        llm_counts = {"A": 0, "B": 0, "SAME": 0}
        # 对人工标注同时统计 A / B 胜出次数以及「平局 SAME」的数量，便于在总览中
        # 感知数据集中“打分相同”的占比。
        human_counts = {"A": 0, "B": 0, "SAME": 0}
        for d in decisions:
            if d.llm_winner in llm_counts:
                llm_counts[d.llm_winner] += 1
            hw = d.human_winner.upper() if d.human_winner else ""
            if hw in human_counts:
                human_counts[hw] += 1

        if self._debug:
            print("\n[Report] ===== 指标汇总结果（与 task.md 对齐）=====")
            print(f"[Report]  单条打分稳定性 α={alpha:.6f}")
            print(
                "[Report]  成对比较一致性: "
                f"pair-accuracy={pair_acc:.6f}, tie_rate={tie_rate:.6f}"
            )
            print(
                "[Report]  Alt-Test: chi_square={:.6f}, p_value={:.6f}".format(
                    chi_square, p_value
                )
            )
            print(
                "[Report]  胜负分布: "
                f"human A={human_counts.get('A', 0)}, "
                f"B={human_counts.get('B', 0)}, "
                f"SAME={human_counts.get('SAME', 0)}; "
                f"llm   A={llm_counts.get('A', 0)}, "
                f"B={llm_counts.get('B', 0)}, "
                f"SAME={llm_counts.get('SAME', 0)}"
            )

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
        """将 LLM 的结构化评分展开为「细粒度评分单元」并按单元聚合。

        旧实现：以 (sample_id, answer_id) 为单位，把每次打分的 total_score
        作为 Krippendorff’s α 的观测值。

        新实现：以「单次打分中的每一条规则项」为最小单位，即：
        - 每个评分单元对应 (sample_id, answer_id, rule_id)；
        - 对同一评分单元在多次采样下的得分列表计算 α；
        - 得分优先使用 ``check['score']``（0~5 区间），若无则退回到布尔命中
          ``check['hit']``（映射为 0/1）。
        """

        stability: Dict[str, List[float]] = {}

        for answer in answers:
            parsed = answer.parsed
            if parsed is None:
                continue

            for idx, check in enumerate(parsed.checks):
                if not isinstance(check, dict):
                    continue

                # 1) 优先使用新版提示词的区间型得分字段 "score"
                if "score" in check:
                    try:
                        rating = float(check.get("score", 0) or 0)
                    except (TypeError, ValueError):
                        # 非法数值直接跳过该条规则项
                        continue
                # 2) 兼容旧版布尔命中格式 "hit" -> {False, True} -> {0.0, 1.0}
                elif "hit" in check:
                    rating = 1.0 if bool(check.get("hit")) else 0.0
                else:
                    # 既没有 score 也没有 hit，无法形成可比较的区间得分
                    continue

                # 规则项标识：优先 rule_id，其次 name / rule，最后退回到索引
                rule_id = (
                    check.get("rule_id")
                    or check.get("name")
                    or check.get("rule")
                    or str(idx)
                )

                unit_id = f"{answer.sample_id}{answer.answer_id}:{rule_id}"
                stability.setdefault(unit_id, []).append(rating)

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
            f"B={report.human_winner_counts.get('B', 0)}, "
            f"SAME={report.human_winner_counts.get('SAME', 0)}",
            "llm_winner_dist:   "
            f"A={report.llm_winner_counts.get('A', 0)}, "
            f"B={report.llm_winner_counts.get('B', 0)}, "
            f"SAME={report.llm_winner_counts.get('SAME', 0)}",
        ]
        return "\n".join(lines)
