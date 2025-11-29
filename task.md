# 任务说明

我要实践LLM-as-judge，探索出一组能够稳定打分的prompt、temperature、topK和topP的设置。 
我有一个初始的prompt该会要求评判模型输出多个维度的评分，以及top K和topP的初始数值。
评判模型对每个回答会给出多个子维度的分值，该具体的维度和打分定义由init_prompt.py中定义，根据prompt的不同可能会有变化。
一条query对应两条answer，这两条answer在打分时是相对分值，因此不同query的answer之间的分值不具备可比性。

我的金标准测试数据的形式如下：(query, ..., a_answer1, b_answer2, winner)。
其中 "query, ..., a_answer1, b_answer2" 分别代表 "用户提问, ...其他数据列..., 回答a, 回答b"。
其中 winner 是**人工的金标准结果**，代表着两条answer中的好坏。

要实现的评估标准方法需要包括： 
（1）单条打分稳定性：Krippendorff’s α 系数（Krippendorff’s Alpha, α），通过把“多次运行同一个 LLM来评价一个answer”视为多个“标注者的结果”，来用 α 度量模型的自一致性。 
（2）成对比较一致性：对一条query的两条answer的llm打分，计算各个维度打分总和，查看得分高低的趋势是否与金标winner数据一致。 
（3）总体数据一致性：替代标注者检验（Alternative Annotator Test, Alt-Test），将 LLM 的评分与人类标注者在数据总集上的winner分布进行比较，证明 LLM 在统计意义上“与人工标注的winner无显著差别”。 

在循环遍历多轮数据的过程中，需要基于各个指标的结果，优化： （1）temperature、topK和topP的数值； （2）prompt文本（你可以调用教师模型chatgpt的api来使用教师模型修改prompt文本）。 

优化的目标为：最大化单条打分稳定性、最大化成对比较一致性、最大化总体数据一致性。 

先给一个整体图景，然后拆成三块：**度量怎么算 → 实验怎么跑 → prompt / 解码参数怎么优化**。要写代码实现的话，这个可以直接当规格书用。
整体思路可以拆成三块：**数据与打分流程设计 → 指标计算 → 搜索与优化策略**。下面直接给你一个可落地的实验框架（不写代码版）。

---

## 一、基础设定与打分流程

1. **数据形式**

   * 每条样本：`(query, ..., a_answer, b_answer, winner)`
   * `winner ∈ {A, B}` 为人工金标。
   * 评判模型的输出：对 `a_answer`、`b_answer`，分别给出多个子维度分数（由 `init_prompt.py` 定义）。

2. **LLM 打分接口**

   * 输入：`(query, answer, 当前 judge-prompt, temperature, topK, topP)`
   * 输出：结构化 JSON（各维度分数 + 总分、必要时 LLm 产生的“推荐 winner”）。

3. **多次运行 = 多“标注者”**

   * 对同一 `(query, answer)`，在固定配置下重复调用 N 次（例如 5–10 次），视作 N 个“标注者”。

---

## 二、指标设计与计算方式

### 1. 单条打分稳定性：Krippendorff’s α

**目标**：同一 `(query, answer)` 在多次采样下打分是否“自洽”。

思路：

1. 对每个配置（prompt + temp + topK + topP）：

   * 对所有 `(query, answer)` 重复调用 N 次，得到 `N × 样本数 × 各维度` 的打分。
2. 对每一个维度单独计算 Krippendorff’s α：

   * 将每次运行视为一位“标注者”，同一 `(query, answer)` 为同一“样本”。
   * 数值型维度用区间/顺序型 α。
3. 得到：

   * 各维度 α
   * 各维度加权平均 α 作为**总体稳定性指标**。

> 优先提升：最低的维度 α（通常是主观性最强的维度）。

---

### 2. 成对比较一致性：与 winner 的方向是否一致

**目标**：在每条 query 内，LLM 认为 “A 比 B 好” 的趋势是否与人工 `winner` 一致。

1. 对每条样本，在每次运行中：

   * 计算总分：`score_A = sum(dim_scores_A)`，`score_B = sum(dim_scores_B)`。
   * 决策：

     * 如果 `score_A > score_B` → `llm_winner = A`
     * 如果 `score_B > score_A` → `llm_winner = B`
     * 如果相等，可以记为“平局”或随机 / 视为不一致（最好单独统计平局率）。

2. 指标：

   * **成对一致准确率**：
     [
     \text{Acc}_{pair} = \frac{#(llm_winner = winner)}{#\text{有明确胜负的样本}}
     ]
   * 也可以：

     * 计算**各维度单独总分**的 pair-accuracy（帮助发现哪个维度更对齐）。
     * 统计**平局率**，作为“模型不确定性”的参考。

3. 为了平滑：

   * 可以对多次运行的结果做**多数投票**：

     * 每条样本上汇总 N 次运行的 `llm_winner`，取出现次数最多的作为最终 `llm_winner` 再和 `winner` 对比。
   * 也可同时看：

     * 单次运行平均的 Acc
     * 多数投票 Acc（更接近“集成判断”的上限）。

---

### 3. 总体数据一致性：Alt-Test（替代标注者检验）

**目标**：在**全部样本层面**，LLM 的“胜负分布”是否与人工无显著差别。

1. 对每条样本生成 LLM 的最终决策：

   * 推荐 `llm_winner ∈ {A, B}`（可以来自：

     * 总分对比，或
     * prompt 中让模型输出“更推荐哪一个”）。

2. 统计两个分布：

   * 人工：`P_human(A胜)`, `P_human(B胜)`（由 `winner` 直接估计）。
   * LLM：`P_llm(A胜)`, `P_llm(B胜)`（由 `llm_winner` 估计）。

3. 做显著性检验：

   * 用例如 **卡方检验 / 二项检验 / McNemar 检验**，检验

     > “LLM 的胜负分布与人工是否存在显著差异”。
   * 得到：p-value、效应量（如差值、KL 散度等）。
   * Alt-Test 通过：

     * p-value 较大（例如 > 0.05），且
     * 效应量较小（统计分布接近）。

---

## 三、搜索与优化策略

### 1. 搜索空间

* **超参维度**：

  * temperature：例如 [0.0, 0.2, 0.4, 0.7]
  * topK：例如 [10, 20, 40, 0/None]
  * topP：例如 [0.5, 0.8, 1.0]
* **prompt 维度**：

  * 初始 prompt 来自 `init_prompt.py`；
  * 通过“教师模型”生成改写版本（例如增强维度定义、增加评分示例、规范输出格式等）。

### 2. 评价函数（多目标）

可以定义一个综合评分函数，对每个配置给出一个标量，便于搜索：

[
\text{Score(config)}
= w_1 \cdot \bar{\alpha}

* w_2 \cdot \text{Acc}_{pair}
* w_3 \cdot f(\text{Alt-Test})
  ]

其中：

* (\bar{\alpha})：各维度 α 的平均或加权平均；
* (\text{Acc}_{pair})：成对一致性准确率；
* (f(\text{Alt-Test}))：将 Alt-Test 的结果映射成 [0,1] 分数，例如：

  * p-value 越大、分布差异越小 → 分数越高；
* (w_1, w_2, w_3) 按你对三项目标的优先级设定。

如不想太复杂，可以先用规则式：

1. 先筛掉：Alt-Test 明显不通过的配置（p-value 极小或分布差距大）；
2. 在剩下的中，按 `Acc_pair` 排序；
3. 若相近，再看 `\bar{\alpha}` 决定谁更优。

### 3. 搜索策略

1. **粗粒度超参搜索（warm-up）**

   * 固定 prompt（原始版本），在较大网格上扫描 temperature, topK, topP。
   * 保留若干个整体评分最好的超参组（Pareto-frontier）。

2. **prompt 优化**

基于错误分析的 Prompt 迭代 (Prompt Optimization with Teacher)
固定基准参数，专注于提升准确率。这是一个循环过程。

**循环步骤：**
1.  **运行评估**：使用当前 Prompt 和基准参数跑完数据集。
2.  **识别错题（Error Analysis）**：找出那些“模型判错”且“置信度高（多次运行结果一致但错了）”的样本。
3.  **构建优化上下文**：
    * 输入：当前 Prompt + 错题 Case（Query, A, B, 模型打分, 人工金标, 人工理由/Feature）。
    * 教师模型指令：*“作为一个 Prompt 优化专家，请分析为什么 LLM 在这些 Case 上判错了（例如：由于 Prompt 中对‘逻辑性’定义的模糊，导致模型偏向了字数更多的错误答案）。请修改 Prompt 中的维度定义或打分标准，以修正这些错误，同时保持对其他正确样本的兼容性。”*
4.  **生成新 Prompt**：由教师模型（ChatGPT）生成 Prompt V2。
5.  **回归测试**：用 Prompt V2 跑测试集。如果“成对一致性”提升且“稳定性”未显著下降，则保留 V2。

---

## 四、落地时的几个小建议

1. **控制成本**：

   * 每个配置先用较小子集试验，只有在子集指标足够好时，才在全量数据上计算 α 和 Alt-Test。
2. **缓存与复用**：

   * 对同一 `(query, answer, prompt, 超参)` 的调用结果缓存起来，以便之后更改指标计算逻辑时不用重新打分。
3. **维度级诊断**：

   * 对每个维度单独看：α、与 winner 的相关性、在 Alt-Test 下的贡献；
   * 识别“噪声维度”，在下一轮 prompt 里弱化或合并。

---

总结：

* 用**重复采样** + Krippendorff’s α 做“自一致性”评估；
* 用**pair-accuracy** 做“每条 query 的胜负方向是否对齐”；
* 用 **Alt-Test** 检查整体分布是否与人工无显著差异；
* 在此基础上，对 **temperature / topK / topP** 做超参搜索，再借助教师模型**迭代优化 prompt**，以这三类指标为目标做多轮搜索即可。

