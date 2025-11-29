# LLM-as-Judge 评估架构文档

本文档给出基于 `task.md` 和 `init_prompt.py` 的整体评估框架设计，覆盖数据流、关键模块职责以及优化策略，便于后续按模块落地实现。

## 总览

目标：针对 (query, a_answer, b_answer, winner) 数据，利用 LLM 作为评估者，优化 prompt 与解码参数，使得：

- **单条打分稳定性**：Krippendorff's α 尽可能高。
- **成对比较一致性**：LLM 判定的胜负方向与人工 `winner` 一致率高。
- **总体数据一致性**：Alt-Test 显示 LLM 的胜负分布与人工无显著差异。

评估流程分为三层：数据与打分 → 指标计算 → 搜索与优化。

## 数据与输入拼装层

| 组件 | 职责 | 关键输入/输出 |
| --- | --- | --- |
| 数据加载器 | 读取 `data.csv`，解析 `(query, modules_block, a_answer, b_answer, winner)`，支持采样/分片。 | 输入：CSV 路径；输出：样本列表或 DataFrame。 |
| 输入模板器 | 将 `query` 和 `last_answer_phone` 拼接成对话文本；将 `data/suggest/rag` 列格式化为 `modules_block`。 | 输入：单条样本；输出：`input_data`、`modules_block`。 |
| Prompt 管理 | 维护 `GROUND_PROMPT_TPL`、`STRUCT_PROMPT_TPL` 以及教师模型生成的变体，支持版本化和回溯。 | 输入：模板名称+变量；输出：填充后的 prompt 文本。 |

## 打分执行层

| 组件 | 职责 | 关键要点 |
| --- | --- | --- |
| 采样控制器 | 管理 temperature、topK、topP、重复次数 N；在同一配置下对每个 `(query, answer)` 重复调用，视为不同“标注者”。 | 支持可重复随机种子、批量并发、调用缓存。 |
| LLM 评分器 | 使用 Ground / Structure 模板对 A、B 两个答案独立打分，输出结构化 JSON。 | 解析 JSON，收集各维度分数与 `confidence`。 |
| 结果缓存 | 以 `(prompt_version, temperature, topK, topP, query_id, answer_id, run_idx)` 为键存储原始响应，避免重复调用。 | 提供重放与回归评估能力。 |

## 指标计算层

| 指标 | 说明 | 依赖数据 |
| --- | --- | --- |
| Krippendorff's α | 将同一 `(query, answer)` 的 N 次分数视为多标注者，按维度计算 α；支持区间/顺序型 α。 | 原始维度分数矩阵。 |
| 成对一致性 Acc | 对每次运行求 `score_A = sum(dim_scores_A)`、`score_B = sum(dim_scores_B)`；统计 `llm_winner` 与人工 `winner` 的一致率，可选多数投票。 | 每次运行的维度分数与 `winner`。 |
| 平局率 | 统计 `score_A == score_B` 的比例，用于衡量不确定性。 | 成对总分。 |
| Alt-Test | 基于最终 `llm_winner` 分布与人工分布做卡方/二项/McNemar 检验，输出 p-value 与效应量。 | 汇总的胜负分布。 |
| 诊断报告 | 输出维度级 α、pair-acc、误差案例，驱动 prompt 调优。 | 上述全部指标与样本级记录。 |

## 搜索与优化层

| 步骤 | 内容 | 产出 |
| --- | --- | --- |
| 粗粒度超参搜索 | 固定初始 prompt，在 temperature、topK、topP 网格上跑小样本集，筛选 Pareto 前沿。 | 若干候选配置。 |
| Prompt 迭代 | 对错误案例构建上下文，调用教师模型生成新 prompt；回归测试后保留提升版。 | Prompt 版本库。 |
| 精细化评估 | 对候选配置在全量数据上计算 α、Acc、Alt-Test；可定义综合得分 `Score = w1*α + w2*Acc + w3*f(Alt-Test)` 进行排序。 | 最优/次优配置清单。 |
| 成本与缓存策略 | 先小样本评估再全量；严格缓存 `(prompt, 超参, 样本)` 调用结果；按维度记录噪声指标。 | 更低调用成本与可复现性。 |

## 模块间数据流

1. **数据加载器** 读取 CSV，传递样本给 **输入模板器**。
2. 模板器生成 `input_data`、`modules_block`，并通过 **Prompt 管理** 生成具体 prompt。
3. **采样控制器** 为每个配置驱动 **LLM 评分器** 对 A/B 答案进行 N 次调用，结果写入 **结果缓存**。
4. **指标计算层** 读取缓存生成的分数，计算 α、pair-acc、Alt-Test 等，并生成 **诊断报告**。
5. **搜索与优化层** 基于指标选择/生成新的 prompt 或解码参数，进入下一轮循环。

## 落地注意事项

- 使用结构化 JSON 响应，解析失败需重试并记录异常样本。
- 对于 Ground / Structure 两个裁判可独立计算指标，也可加权合成最终得分。
- 统计输出应包含：各维度 α、pair-acc、平局率、Alt-Test p 值、示例错误案例列表。
- 日志与缓存应方便回溯每个配置的原始 LLM 输出，便于 prompt 诊断。

