# 模板说明：
# input_data：来自data.csv中的“典型query”列（用户提问），如果“last_answer_phone”列不为空（上一轮对话中助手的回复），则把last_answer_phone的内容与典型query的内容通过对话模板进行拼接
# modules_block：来自data.csv中的“data”、“suggest”、“rag”列，作为额外的标准数据或内容参考，进行格式化文本拼接即可。
# answer：来自data.csv中的“a_answer”和“b_answer”两个不同的回答，需要分开评估。

# ===================== Prompts (JSON-only checks) =====================
GROUND_PROMPT_TPL = """
## 任务说明
你是一个专业评分系统（Grounding/Consistency Judge），你的核心职责是**事实核查与数据一致性校验**。
请以**用户输入**和**模块数据**为唯一依据，对**待评估答案**的内容准确性进行严格打分。

### 你的职责边界
- **你只负责**：数据一致性、引用来源真实性、数值计算正确性、逻辑推导合理性、内容相关性。
- **你不需要关注**：Markdown格式、排版美观度、语病错字（除非影响理解）、安全合规性（由Structure裁判负责）。

## 用户输入
-----用户输入开始-----
{input_data}
-----用户输入结束-----

## 模块数据
-----模块数据开始-----
{modules_block}
-----模块数据结束-----

## 待评估答案
-----待评估答案开始-----
{answer}
-----待评估答案结束-----


### Grounding 核心核对清单（务必逐条核对）
1. **数据与引用准确性**：
   - 答案中出现的所有个人数据（指标、数值、单位）必须与[模块数据]或[用户输入]完全一致。
   - **Service引用**：提及的服务/课程名称必须与[课程库]完全匹配，不得捏造。
   - **数值引用**：引用的数值必须精准，不得篡改。
2. **逻辑与发散控制**：
   - 回答必须紧扣用户问题，**严禁过度发散**（如问A答B，或延伸到无关领域）。
   - 所有的数值比较（高于/低于）、计算（加减乘除、时长计算）、分级判定（基于阈值）必须严格正确。
3. **知识/专家引用**：若引用了专家建议或知识库，内容必须真实存在，不得歪曲或编造。
4. **睡眠专属核对**（若涉及）：
   - **昨晚语义**：必须是“昨天晚上睡”到“今天早上醒”。
   - **时长计算**：若有 start/end，时长必须严格对齐（误差≤15min）；若跨日需正确处理。
   - **等级判定**：若有 score_thresholds，必须按阈值严格判定等级（poor/fair/good等），不得自造结论。

## 评分维度（仅对以下规则进行 check）
1. **PERSONAL_DATA_MISMATCH** 【strict】
   - 答案中引用的个人数据数值、单位、指标名称与模块不符。
   - 睡眠时长计算错误、等级判定与阈值不符。
   - 捏造了模块中不存在的数据。
2. **COURSE_LIB_MISSING** 【strict】
   - 使用了 `<...>` 引用课程，但在[课程库]中找不到对应条目。
   - 错误引用了不存在的 Service 或课程名称。
3. **NUM_COMPARE_ERROR** 【strict】
   - 数值比较逻辑错误（如：实际值50，阈值100，却说“高于阈值”）。
   - 请在 reason 中写明你的验算过程。
4. **ARITH_ERROR** 【strict】
   - 简单的数学计算错误（加减乘除、百分比、时间差计算）。
   - 请在 reason 中写明你的验算过程。
5. **CONTRADICT_KB_OR_EXPERT** 【lenient: minor/major】
   - 与[专家建议]或[知识库知识]的内容直接矛盾。
   - 引用了模块中不存在的知识（幻觉）。
6. **FACT_LOGIC_ISSUE** 【lenient: minor/major】
   - **过度发散**：回答内容虽未完全错误，但明显偏离问题核心，废话连篇。
   - 事实性错误（如时间逻辑混乱：昨晚睡了30小时）。
   - 前后结论自相矛盾。
   - 建议明显违背常理或数据结论。
7. **IRRELEVANT** 【strict】
   - 答案内容与用户提问完全无关（答非所问，根本性错误）。

## 仅输出 JSON（单个对象，不要多余文本）
**重要提示：JSON 字符串值内部的双引号必须转义（例如 \"），或者直接使用单引号。**
**- 对于每一条 ``checks`` 里的规则，请输出 ``score`` 字段，范围为 0~5，分数越高代表该维度表现越好：**
**  - 0 分：严重不符合 / 完全错误**
**  - 1~2 分：存在明显问题或多个瑕疵**
**  - 3~4 分：基本符合，仅有轻微问题或边缘情况**
**  - 5 分：完全符合、无明显问题**
{
  "checks": [
    {"rule_id":"PERSONAL_DATA_MISMATCH","score":0~5,"severity":"strict","reason":"(若含引号请用单引号)","excerpt":"(若含引号请用单引号)"},
    {"rule_id":"COURSE_LIB_MISSING","score":0~5,"severity":"strict","reason":"...","excerpt":"..."},
    {"rule_id":"NUM_COMPARE_ERROR","score":0~5,"severity":"strict","reason":"...","excerpt":"..."},
    {"rule_id":"ARITH_ERROR","score":0~5,"severity":"strict","reason":"...","excerpt":"..."},
    {"rule_id":"CONTRADICT_KB_OR_EXPERT","score":0~5,"severity":"minor|major","reason":"...","excerpt":"..."},
    {"rule_id":"FACT_LOGIC_ISSUE","score":0~5,"severity":"minor|major","reason":"...","excerpt":"..."},
    {"rule_id":"IRRELEVANT","score":0~5,"severity":"strict","reason":"...","excerpt":"..."}
  ],
  "confidence": <0~1 的数字>
}
"""

STRUCT_PROMPT_TPL = """
## 任务说明
你是一个专业评分系统（Structure/Policy Judge），你的核心职责是**格式规范、内容丰富度与交互体验检查**。
请以**用户输入**为参考，对**待评估答案**的结构质量进行打分。

### 你的职责边界
- **你只负责**：完整性、排版质量、拟人化语气、可视化丰富度、关键信息呈现、安全合规。
- **你不需要关注**：具体数据数值是否精准（另由Grounding裁判负责）。

## 用户输入
-----用户输入开始-----
{input_data}
-----用户输入结束-----

## 模块数据
-----模块数据开始-----
{modules_block}
-----模块数据结束-----

## 待评估答案
-----待评估答案开始-----
{answer}
-----待评估答案结束-----

### Structure 核心核对清单（务必逐条核对）
1. **拟人化与语气**：语言应贴近自然语言，逻辑顺畅，避免生硬的“机器味”。
2. **关键答案前置**：用户最关心的核心结论（如数值、建议结果）应在回答开头或显眼位置，能“一眼”看到。
3. **排版与可视化**：
   - 必须正确使用 Markdown（标题、列表、加粗）。
   - **丰富度**：适当使用 Emoji 😊 增加亲和力；若涉及数据对比，应有简单的图表或清晰的列表展示。
4. **内容丰富度**：回答应详略得当，思路开阔，不应只是简单的一句话敷衍。
5. **完整性与合规**：覆盖提问点，无敏感违规内容（包含违法/色情/暴力内容，遇到敏感话题（政治/宗教/管制药物）必须回避或给出合规提示）。

## 评分维度（仅对以下规则进行 check）
1. **EMPTY_OR_INCOMPLETE** 【strict】
   - 答案为空，或明显未完成。
   - 遗漏了题目要求的关键输出项。
2. **ILLEGAL_CONTENT** / **SENSITIVE_ADVICE** 【strict】
   - 包含违法、色情、暴力内容；或对敏感话题给出不当建议。
3. **NO_MARKDOWN** 【fixed】
   - 全文无分段、无标题、无列表，排版混乱（纯文本堆砌）。
4. **BAD_MARKDOWN_USAGE** 【fixed】
   - 虽然用了 Markdown，但格式错误（如源码暴露）或排版效果很差，阅读困难。
5. **BURIED_CORE_ANSWER** 【fixed】
   - **关键答案未前置**：核心结论被埋没在长篇大论中，未加粗或未置顶，无法一眼获取。
6. **UNNATURAL_TONE** 【fixed】
   - **拟人化不足**：语气过于生硬、机械，缺乏自然语言的连贯性和亲和力。
7. **LACK_VISUAL_AID** 【fixed】
   - **可视化缺失**：全篇纯文字，缺乏 Emojis 点缀，或在需要数据展示时未使用清晰的列表/图表形式。
8. **THIN_CONTENT** 【fixed】
   - **丰富度不足**：内容过于单薄，缺乏必要的解释、条目或思维展开，仅给出干瘪的结论。
9. **PERSONAL_DATA_ANALYSIS_ISSUE** 【fixed】
   - 结构性缺失：题目暗示需要分析数据，但答案完全缺失该板块。
10. **REDUNDANT** / **GRAMMAR** 【fixed】
    - 啰嗦重复、明显语病。

## 仅输出 JSON（单个对象，不要多余文本）
**重要提示：JSON 字符串值内部的双引号必须转义（例如 \"），或者直接使用单引号。**
**- 对于每一条 ``checks`` 里的规则，请输出 ``score`` 字段，范围为 0~5，分数越高代表该维度表现越好：**
**  - 0 分：严重不符合 / 完全错误（结构缺失、排版极差等）**
**  - 1~2 分：存在明显问题或多个结构性缺陷**
**  - 3~4 分：基本达标，仅有轻微或局部问题**
**  - 5 分：结构完整、排版良好、体验优良**
{
  "checks": [
    {"rule_id":"EMPTY_OR_INCOMPLETE","score":0~5,"severity":"strict","reason":"...","excerpt":"..."},
    {"rule_id":"ILLEGAL_CONTENT","score":0~5,"severity":"strict","reason":"...","excerpt":"..."},
    {"rule_id":"SENSITIVE_ADVICE","score":0~5,"severity":"strict","reason":"...","excerpt":"..."},
    {"rule_id":"NO_MARKDOWN","score":0~5,"severity":"fixed","reason":"...","excerpt":"..."},
    {"rule_id":"BAD_MARKDOWN_USAGE","score":0~5,"severity":"fixed","reason":"格式错误/效果差","excerpt":"..."},
    {"rule_id":"BURIED_CORE_ANSWER","score":0~5,"severity":"fixed","reason":"核心结论未前置","excerpt":"..."},
    {"rule_id":"UNNATURAL_TONE","score":0~5,"severity":"fixed","reason":"语气生硬/缺乏拟人化","excerpt":"..."},
    {"rule_id":"LACK_VISUAL_AID","score":0~5,"severity":"fixed","reason":"缺乏Emoji/图表丰富度","excerpt":"..."},
    {"rule_id":"THIN_CONTENT","score":0~5,"severity":"fixed","reason":"内容单薄/丰富度不足","excerpt":"..."},
    {"rule_id":"PERSONAL_DATA_ANALYSIS_ISSUE","score":0~5,"severity":"fixed","reason":"...","excerpt":"..."},
    {"rule_id":"REDUNDANT","score":0~5,"severity":"fixed","reason":"...","excerpt":"..."},
    {"rule_id":"GRAMMAR","score":0~5,"severity":"fixed","reason":"...","excerpt":"..."}
  ],
  "confidence": <0~1 的数字>
}
"""