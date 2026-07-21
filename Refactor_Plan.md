# ChainUQ 全面重构计划（实时更新）

> 最后更新：2026-07-15 05:32 EDT
> 当前阶段：**按用户要求暂停；P19 仅完成 Llama 正向审计，尚未运行 Mistral**
> 投稿目标：AAAI-27 Main Technical Track（摘要截止 2026-07-21 AoE，全文截止 2026-07-28 AoE）

## 0. 本轮目标与不可妥协项

- 论文故事、研究问题、方法和实验均允许推倒重做；不以保住现有 `ChainUQ` head 或 judge 为目标。
- 研究主轴必须落在 **uncertainty + hallucination**，且要形成一个可证伪、可被实验直接支持的中心命题。
- 优先复用当前的数据加载、生成、缓存、模型 wrapper、评估指标和作业脚本；旧 head、R-Log 和 judge 仅在新假设确实需要时保留。
- 运行根目录固定为仓库内的 `popllm` 符号链接：
  - 模型：`/home/dy23a.fsu/popllm/ChainUQ/popllm/models`
  - 数据：`/home/dy23a.fsu/popllm/ChainUQ/popllm/datasets`
  - 生成缓存：`/home/dy23a.fsu/popllm/ChainUQ/popllm/cached_features`
  - 新实验建议：`/home/dy23a.fsu/popllm/ChainUQ/popllm/results/refactor`
- 本机硬件：NVIDIA B200 183GB；第一轮以小样本验证关键假设，方向确定后再扩大模型和数据矩阵。
- **种子策略更新（用户要求）**：后续所有探索与正式实验只使用固定 seed=2026，不再运行多种子；用更多 backbone、dataset、bootstrap confidence interval 和 paired test 检验稳定性。

## 1. 实时进度

| 状态 | 工作项 | 当前结论 / 产物 |
|---|---|---|
| ✅ | 阅读 LaTeX 主文、附录、表格和三份审稿意见 | 旧稿的主要问题不是写法，而是贡献不可识别：复杂 head 像 ad-hoc patchwork；外部 24B judge 带来不公平比较/潜在标签代理；结构化 CoT 与 atomic claim 假设限制泛化。 |
| ✅ | 审计代码和数据基础设施 | 已确认完整的 Generate → Judge → Cleanup → Train → Evaluate 链路、9 个模型目录、14 个数据目录以及约 82GB 已清理 feature cache；这些基础设施可复用。 |
| ✅ | 核验 GPU 与路径 | B200 可用且空闲；`popllm` 指向 `/blue/fsu-compsci-dept/dy23a.fsu/popllm`。当前 `config.py` 默认路径错误，待方向确定后统一修复。 |
| ✅ | 第一轮 2024–2026 文献查重 | CoT-UQ、trajectory entropy、step UHead、因果扰动、adaptive compute、conformal early exit、RAG factuality/UQ 均已拥挤；不能只把旧方法换一个动态 head。 |
| ✅ | 现有缓存的诊断性预实验 | 汇总 149,321 个带 step label 的样本：答案正确但至少一步被判错占 **12.96%**；答案错误但所有步骤被判对占 **25.41%**；两类轴合计错位 **38.37%**。仅用“正确步骤比例”预测答案正确性的 pooled AUROC 只有 **0.5593**（Spearman 0.1305）。 |
| ✅ | 候选方向排序和最小验证 | 已形成 3 个主候选；P1 已在 Llama-3.1-8B 与 Mistral-7B 上完成 3 seeds 复现。 |
| ✅ | A/B/C 小样本方向选择 | A 在两个 backbone 上稳定通过；B 的首错命题失败；C 跨 backbone 不稳定且成本较高。当前实证推荐 A。 |
| ❌ | TypeUQ 强 baseline 对比 | TypeUQ conditional 在 ID/OOD 均未超过 direct four-way / separate experts；按用户要求停止该方向。 |
| ❌ | 新方向：CF-UQ v2 | Mistral 强、Llama 失败；固定 evidence-dependency score 在 Llama 也仅 0.558 AUROC。正式停止 C。 |
| ❌ | Reasoning Amplification UQ | Llama 提升、Mistral 退化，未通过跨模型 gate。 |
| ❌ | Evidence-Bottleneck UQ | reasoning-aligned 检索将 gold-title recall 提至 85.9%，但 AUROC 仍低于 full baseline；停止。 |
| ❌ | CompatibilityUQ | OOD Macro-F1 有局部改善，但 ID/OOD all-good AUROC 均弱于强 baseline；停止。 |
| ❌ | 多域鲁棒 hallucination UQ | type-aware GroupDRO 仅改善 worst AUROC、明显损害 mean AUROC，未同时超过 balanced ERM；停止。 |
| ❌ | Domain-relative UQ | 无标签目标域 normalization 未超过 global balanced ERM，且 calibration 更差；停止。 |
| ❌ | 多轨迹 lexical path UQ | 固定组合仅 +0.008 AUROC，paired bootstrap CI 跨 0；learned stack 退化，停止。 |
| ❌ | Grounded Consensus UQ | 模型 citation compliance 仅约 10%，grounding AUROC 低于随机，组合与 learned stack 均弱于 self-consistency。 |
| ❌ | Contrastive Answer Margin UQ | 相对 weighted vote 有显著增益，但未超过最强 direct detached likelihood；融合仅 +0.003 AUROC 且 AUPRC 下降。 |
| ❌ | Reasoning-Detached UQ | Llama 强但 Mistral 低于 self-consistency；统一 rank ensemble 也不能同时改善 AUROC/AUPRC。 |
| ❌ | Verified Consensus UQ | modal P(True)、verifier margin/mass 与 learned verifier stack 均未超过修正后的强 baseline；停止。 |
| ❌ | Evidence-Order Invariance UQ | 对 vanilla-8 的跨模型增益成立，但同为 16 次生成时，Llama N=192 的 evidence-order joint=0.7417，低于 repeated-sampling joint=0.7552 与 vanilla-16=0.7576；收益不能排除纯采样预算效应，停止。 |
| ❌ | Contextual Evidence Gain UQ | 去 reasoning-prefix 后 raw context gain AUROC 仅 0.5996；最佳固定组合 0.7280，与 SC 0.7276 持平且 CI 跨 0，未运行第二模型即停止。 |
| 🟡 | Set-Relative Self-Verification UQ | canonical 修正后 Llama verifier mass=0.7583，超过 SC=0.7276、contrastive mass=0.7114、absolute P(True)=0.6153；正在 Mistral 复验，之后补 compute-matched sampling baseline。 |
| ⏳ | 新方法实现、正式基线和全量实验 | 方向确定后开始。 |
| ⏳ | LaTeX 全文重写 | 等中心命题和核心结果站稳后重写，避免围绕未成立的方法先写故事。 |

## 2. 旧 ChainUQ 的结论：哪些保留，哪些停止投入

### 2.1 可保留的工程资产

1. 多模型加载与 vLLM/HF 双后端。
2. HotpotQA、MuSiQue、2Wiki、IIRC、StrategyQA、StepGame、bAbI 等数据适配器。
3. 已对齐的 response / conclusion / reasoning token features，以及 split storage。
4. correctness、AUROC、PRAUC、ECE、Brier、selective risk 等评估代码。
5. 批量 job、cache recovery、日志和结果汇总脚本。

### 2.2 不再作为论文贡献的部分

1. **复杂 lightweight head 本身**：projection、gated pooling、bank、context fusion、residual refinement 的组合缺少单一原理，容易再次被评价为 patchwork。
2. **judge-conditioned R-Log 校准作为主创新**：其增益无法与 24B judge 能力清晰分离，且 baseline access 不对称。
3. **“reasoning consistency 必然改善 answer UQ”这一前提**：现有 149k 样本已经显示两者大幅错位，应该把错位本身变成研究对象。
4. **仅四个 QA 数据集上的 answer correctness**：这不足以覆盖 hallucination 的 intrinsic/extrinsic、factuality/faithfulness、confident hallucination 等类型。

## 3. 最有希望的三个重构方向（等待选择）

### A. TypeUQ：Uncertainty Is Not Hallucination（首选）

**一句话故事**：单一 confidence 把“模型不知道”“输出不受证据支持”“推理过程无效”压成一个数，因此在不同 hallucination 类型间不可迁移；我们将其分解为可校准、可行动的 typed risks。

**中心命题**：hallucination 不是 uncertainty 的同义词。可靠系统应同时估计至少三个条件风险：

1. `answer risk`：最终答案错误概率；
2. `grounding risk`：回答不受给定证据支持的概率；
3. `process risk`：推理链包含无效跃迁/错误步骤的概率。

三者的 disagreement 是需要报告和利用的信号，而不是由 calibrator 抹平的噪声。

**方法草案**：

- 用共享 frozen features + 三个极简 risk heads，禁止再堆复杂模块。
- 构造最小反事实配对（保持问题/表述风格，分别干预 evidence、reasoning step、answer）进行 factor-aware contrastive training，避免探针学习“错误文本长什么样”。
- 输出 risk vector，并用一个显式决策层把不同风险映射到 `answer / abstain / retrieve / verify-reasoning`；judge 可用于离线弱标注，但不进入推理特征。
- 加入 typed calibration / selective risk；若时间允许，给每个 risk 维度做 split-conformal coverage，而不是声称一个总 confidence 有保证。

**新实验主线**：

- `factor identification`：分别预测 correctness、faithfulness、process validity；报告 cross-type / cross-domain transfer。
- `counterfactual sensitivity`：只改变一个风险因素时，对应 head 应变化，其他 head 应保持稳定。
- `confident hallucination`：专门测 CHOKE/false-premise/证据冲突类样本，检验普通 entropy 与 typed risk 的差异。
- `action utility`：相同验证/检索预算下，typed router 是否比 scalar uncertainty 降低 hallucination risk。
- 人工复核小子集，避免新故事再次完全依赖 LLM judge。

**为什么适合 AAAI**：问题定义清晰，能给出 scalar-risk non-identifiability 的形式化论证；方法简单；诊断、方法、决策效用形成完整闭环。它也直接承接 2026 年“uncertainty 对 hallucination 的相关性不稳定”的新发现，但从 evaluation 推进到 factorized estimation + action。

**主要风险**：FRANQ 已区分 RAG factuality/faithfulness，几何 taxonomy 也区分 hallucination 类型；必须用“reasoning risk + factorial counterfactual benchmark + downstream action”建立明显边界。

**当前证据**：149,321 个现有样本中 answer/process 两轴有 38.37% 错位，已否定旧稿的单轴直觉，支持开展 typed-risk 建模。

### B. HazardUQ：Where Did Reasoning First Go Wrong?（次选）

**一句话故事**：response-level confidence 发现错误太晚；把每一步视为 hazard event，在线估计“首个不可恢复错误”发生位置，并在错误扩散前触发验证或重生成。

**中心命题**：终局错误概率应由 step-wise error hazard 累积得到；首错位置和可恢复性比整条 CoT 的 pooled score 更有解释力。

**方法草案**：从每一步结束位置的 hidden/logit innovation 估计 hazard；通过 survival likelihood 训练；输出 answer survival probability 和 first-error localization；无 judge 推理开销，judge/规则仅用于训练标签。

**新实验主线**：首错定位、prefix-only early warning、不同观察预算下 AUROC、触发局部修正后的准确率—token Pareto、跨 reasoning length/domain 迁移。

**优点**：保留 ChainUQ 的 step cache，形式化比旧 head 干净，结果直观。

**风险**：2026 年已有 uncertainty trajectory、step UHead、mid-reasoning correctness、adaptive sampling；必须把 `first-error survival + recoverability + intervention` 做扎实，否则容易被视为换损失函数。

### C. CF-UQ：Counterfactual Evidence Sensitivity for Confident Hallucinations（第三选择）

**一句话故事**：普通 uncertainty 漏掉“模型其实知道答案但被轻微提示诱导后自信答错”的 CHOKE 类错误；可信度应测量答案对因果相关证据的敏感性，而非只测输出分布熵。

**中心命题**：对证据支持、删除和冲突的最小干预所产生的 confidence response curve，可以区分 knowledge gap 与 evidence override / shortcut hallucination。

**方法草案**：自动生成受控 evidence interventions；提取 answer logit/hidden-state sensitivity；以 invariance + monotonicity objective 学习一个轻量 detector；推理时可选择 1–2 个低成本干预或蒸馏成单次 head。

**新实验主线**：CHOKE、false-premise、Hotpot/2Wiki evidence swap；检测 AUROC；对 paraphrase 的不变性；跨模型/领域；干预次数—性能曲线。

**优点**：hallucination 味道最强，反事实实验比普通 correlation 更有说服力。

**风险**：与 FACT-E、causal faithfulness、counterfactual CoT 文献相邻；新数据构造和严谨 controls 工作量最大，AAAI-27 时间风险最高。

## 4. 当前排序与决策建议

| 方向 | Novelty | AAAI 完整故事 | 13 天可行性 | 与现有资产兼容 | 总体建议 |
|---|---:|---:|---:|---:|---|
| A. TypeUQ | 4.5/5 | 4.7/5 | 4.0/5 | 4.5/5 | **首选；建议立即押注** |
| B. HazardUQ | 3.7/5 | 4.2/5 | 3.8/5 | 4.8/5 | 保守次选，需避免撞 trajectory/UHead |
| C. CF-UQ | 4.3/5 | 4.4/5 | 2.6/5 | 3.5/5 | 更适合较宽松周期；AAAI-27 高风险 |

实验前建议为 A+C；完成三方向 pilot 后更新为：**主方向只押 A/TypeUQ**。C 的 supporting-evidence deletion 在 Llama 上失败，不能作为当前方法的必要组成；后续最多把更严格的反事实配对作为诊断性评估，不把它写成已成立的核心方法。

## 5. 方向 A 的最小可行实验（正在执行）

### Pilot P0：旧前提诊断（已完成）

- 数据：当前全部已生成 cache 的 index metadata，4 个 backbone、7 个数据集、149,321 个样本。
- 结果：
  - `correct answer + flawed chain`：12.96%
  - `wrong answer + all steps judged valid`：25.41%
  - answer/process disagreement：38.37%
  - reasoning-correct fraction → answer correctness：AUROC 0.5593，AP 0.6520，Spearman 0.1305
- 含义：process consistency 既不是 answer correctness，也不是其稳定代理；一个 scalar 目标会隐藏结构性错位。
- 限制：process label 来自旧 judge，因此 P0 是方向筛选证据，不作为最终论文的唯一证据。

### Pilot P1：双风险可识别性（已完成）

- 在 Llama-3.1-8B 与 Mistral-7B / HotpotQA 上用现有 1.3k–5k 训练样本训练极简 probe，各跑 3 seeds。
- 比较：`shared_scalar` 强制 answer/process 两个输出共享同一个一维风险方向；`typed_linear` 为两个目标学习独立线性方向。
- 输入：结论 frozen feature、推理 token feature 的简单 mean/max summary，不使用 judge verdict 作为 feature。
- 输出：answer AUROC、process AUROC、两轴 exact match、四象限 macro-F1、ECE/Brier。
- 结果（test，mean ± sample std over 3 seeds）：

| Backbone | Method | Answer AUROC | Process AUROC | Mean AUROC | 4-quadrant Macro-F1 |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | shared scalar | 0.5827 ± 0.1298 | 0.7758 ± 0.0380 | 0.6793 ± 0.0565 | 0.2537 ± 0.0169 |
| Llama-3.1-8B | typed linear | **0.7209 ± 0.0018** | **0.7935 ± 0.0020** | **0.7572 ± 0.0018** | **0.3426 ± 0.0112** |
| Mistral-7B | shared scalar | 0.5672 ± 0.1203 | 0.7798 ± 0.0411 | 0.6735 ± 0.0600 | 0.2474 ± 0.0374 |
| Mistral-7B | typed linear | **0.6699 ± 0.0080** | **0.7922 ± 0.0113** | **0.7311 ± 0.0096** | **0.3835 ± 0.0114** |

- 判断：P1 **通过 go 条件**。独立风险方向在两个 backbone 上分别提高 mean AUROC 0.0779 / 0.0576，并显著提高四象限 macro-F1；一维瓶颈跨 seed 极不稳定，而 typed linear 很稳定。这支持“不能把 answer/process 压成一个 scalar”的核心诊断。
- 限制：当前只区分 answer/process 两轴，grounding 轴和反事实选择性尚未验证；Brier 尚未单独校准，当前只把 AUROC/Macro-F1 作为 P1 决策证据。
- 可复现脚本：`analysis/pilot_typed_risk.py`；原始 JSON 位于 `popllm/results/refactor/pilot_typed_risk*.json`。

### Pilot P2：最小反事实（待 P1 后）

- 从 HotpotQA/2Wiki 抽取 100–300 个正确样本，分别做 answer swap、supporting-evidence deletion、reasoning-step corruption。
- 检验 risk vector 的选择性响应：目标维度变化显著、非目标维度相对稳定。
- Go 条件：相对 scalar baseline 有稳定的 intervention selectivity，并至少在两个 backbone 上复现。

### 三方向探索的统一决策标准（2026-07-15 新增）

为了避免用不可比较的单项指标“挑赢家”，A/B/C 均按以下五项判定：

1. **核心假设是否被直接支持**：不是只看最终 AUROC，而是看与论文命题一一对应的干预/定位结果。
2. **效应量**：相对最自然且信息权限一致的 baseline 有多大提升。
3. **跨 backbone 稳定性**：至少 Llama-3.1-8B 与 Mistral-7B 方向一致。
4. **推理成本**：单次 head、逐 step head、或每样本额外 counterfactual forward 的成本差异。
5. **论文空间**：结合 2024–2026 相邻工作，判断正结果能否形成 AAAI 级别的新中心命题。

当前运行队列：

- A/TypeUQ：P0/P1 已完成；**GO**。
- B/HazardUQ：step verification 有信号，但 first-error 与 final-answer 命题未成立；**NO-GO（作为整篇故事）**。
- C/CF-UQ：Mistral 有信号、Llama 失败，且需 3 倍 teacher-forcing；**NO-GO（当前 deletion 版本）**。

### Pilot P3：B/HazardUQ（已完成）

设置：使用每个 reasoning step 的简单 mean feature；`hazard_delta` 只看当前 step、与上一步的 feature delta、归一化位置。训练标签为 step validity，推理输入不含 judge verdict。自然 baseline 是 step token maximum log-probability。

| Backbone | Method | Step-valid AUROC | First-error Top-1 | All-valid AUROC | Answer AUROC |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B | token log-prob | 0.6167 | **0.5129** | 0.6863 | 0.5314 |
| Llama-3.1-8B | hazard delta | **0.6892** | 0.5074 | **0.7146** | **0.5434** |
| Mistral-7B | token log-prob | 0.6382 | 0.5601 | 0.7561 | 0.6248 |
| Mistral-7B | current-step linear（最佳 learned variant） | **0.7414** | **0.5636** | **0.8016** | **0.6356** |

结论：

- learned step feature 对 step validity 有稳定增益（+0.073 / +0.103 AUROC），说明缓存足以训练轻量 step verifier。
- 但是首错定位几乎没有超过 token baseline（Llama 甚至下降），与“Where did reasoning first go wrong?”的中心命题不符。
- 累积 survival 对 final answer 的增益只有约 +0.012 / +0.011 AUROC；reasoning hazard 仍不是 answer hallucination 的强代理。
- 因此 B 作为论文主线 **NO-GO**。若降级成 step verifier，创新性又会与已有 step UHead/PRM 工作高度重合。
- 可复现脚本：`analysis/pilot_hazard_uq.py`；结果：`popllm/results/refactor/pilot_hazard_uq*.json`。

### Pilot P4：C/Counterfactual Evidence Sensitivity（已完成）

设置：每个 HotpotQA response 固定原生成 reasoning prefix 和 conclusion，分别在 `full context`、`supporting facts only`、`supporting facts deleted` 下 teacher-force conclusion，得到三点 answer log-likelihood curve。每个 backbone 使用 64 train + 64 test 平衡样本，共 384 条 teacher-forced sequence。

| Backbone | Direct full log-p AUROC | Sensitivity-only AUROC | Full CF curve AUROC | Correct vs incorrect `full-deleted` |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 0.4521 | 0.5264 | 0.5205 | 0.1475 vs 0.1501（无分离） |
| Mistral-7B | 0.7666 | 0.7236 | **0.7949** | 0.3407 vs 0.0377（强分离） |

结论：

- Mistral 的 evidence response curve 很有意思，完整 curve 相对 direct score 提升 +0.0283 AUROC，且正确回答对删除 supporting facts 更敏感。
- 同样协议在 Llama 上接近随机，核心 sensitivity gap 完全不分离。这不是小样本 smoke 的偶然解释：扩大到 64+64 后早期 0.617 的信号消失。
- 每个样本需要三次长 prompt teacher-forcing，相对 direct score 是 3 倍 forward 成本；本次 128 样本/384 sequences 在 B200 上约 34–35 秒（含模型加载与数据准备）。
- C 当前版本不满足跨-backbone稳定性，故 **NO-GO**。未来若重开，应改成语义受控的 contradiction / entity-swap 配对，而不是简单 evidence deletion，并单独研究为什么 Llama/Mistral 响应相反。
- 可复现脚本：`analysis/pilot_counterfactual_uq.py`；结果：`popllm/results/refactor/pilot_counterfactual_uq*.json`。

## 5.1 三方向最终横向结论

| 方向 | 核心实验证据 | 跨模型 | 额外成本 | 核心命题判定 | 最终排序 |
|---|---|---|---|---|---:|
| A. TypeUQ | mean AUROC +0.0779/+0.0576；四象限 Macro-F1 +0.0889/+0.1361 | 稳定 | 单次轻量 head | **GO** | **1** |
| C. CF-UQ | Llama 0.5205；Mistral 0.7949 | 不稳定 | 3× teacher-forcing | 当前版本 NO-GO，但现象值得以后研究 | 2 |
| B. HazardUQ | step AUROC 提升，但首错不提升，answer 仅 +0.01 | step 信号稳定、核心命题不成立 | 每 step 轻量 head | 主线 NO-GO | 3 |

**最终推荐：A/TypeUQ。** 选择它不是因为绝对 AUROC 最大，而是因为它是唯一同时满足“核心命题被直接支持、两个 backbone 方向一致、成本低、能解释旧 ChainUQ 失败”的方案。下一阶段应补齐 grounding 第三轴和反事实 factorial labels，而不是继续扩展现有双轴 linear pilot。

### P5：TypeUQ 强基线闸门（已完成，NO-GO）

旧 P1 的 `shared_scalar` 是诊断 bottleneck，不足以证明方法优于强 baseline。P5 采用相同 frozen features、相同 train/validation/test、相同标签权限，比较：

1. `token_confidence`：不训练的 conclusion/reasoning token log-prob 聚合。
2. `scalar_mlp_matched`：直接监督 `answer_correct AND process_valid`，参数量与 TypeUQ 匹配。
3. `fourway_linear`：直接预测 answer/process 的四个联合状态，参数量高于 TypeUQ。
4. `separate_experts`：分别训练 answer 和 process linear probes，再组合联合概率；这是必须击败或至少持平的强基线。
5. `typeuq_conditional`：显式建模 `P(answer)` 与 `P(process | answer)`，通过概率链式分解得到四类联合风险和两个边缘风险。

评估：HotpotQA ID + MuSiQue/IIRC/2Wiki/StrategyQA/StepGame/bAbI OOD；报告 answer/process/all-good AUROC、四象限 Macro-F1、Brier/ECE、平均 OOD 和 worst-domain。固定 seed=2026。

Go/no-go：

- TypeUQ 必须在 joint-risk 或四象限指标上稳定超过 matched scalar 和 direct four-way baseline，并且 OOD 平均不是只靠单一数据集拉高。
- 对 separate experts 至少不能显著退化；若只与 separate experts 打平且没有 action/calibration 优势，则当前 TypeUQ 不构成方法贡献。
- 若上述条件失败，停止围绕 TypeUQ 继续堆 head，转向新的中心命题。

Llama-3.1-8B 实验结果（seed=2026；HotpotQA train=4,963，ID test=3,672，六个 OOD 各最多 1,000）：

| Method | Params | ID All-good AUROC | ID 4-way Macro-F1 | OOD mean All-good AUROC | OOD mean 4-way Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Matched scalar MLP | 25,219 | 0.7287 | N/A | 0.6380 | N/A |
| Direct four-way linear | 33,620 | **0.7535** | **0.3707** | 0.6929 | **0.2748** |
| Separate experts | 16,810 | 0.7408 | 0.3433 | **0.6961** | 0.2169 |
| TypeUQ conditional | 25,215 | 0.7347 | 0.3283 | 0.6718 | 0.2342 |

结论：TypeUQ conditional 没有击败任一关键强 baseline；ID 和 OOD 的 joint risk、四象限预测均存在明确退化。原 P1 只证明“一维 bottleneck 太弱”，没有证明 typed factorization 是更强方法。**TypeUQ 正式 NO-GO，不再继续扩 head 或扩大矩阵。**

可复现脚本：`analysis/typeuq_baseline_gate.py`；结果：`popllm/results/refactor/typeuq_baseline_gate_llama.json`。

### P6：CF-UQ v2（已完成，NO-GO）

切换理由：上一版 counterfactual 实验把 cached reasoning prefix 固定后再评分 conclusion，但该 prefix 已复述 supporting evidence，导致删除原始 context 后模型仍可从 prefix 读取证据。v2 移除 reasoning prefix，只保留 `Conclusion:`，直接测生成答案在 full/support-only/support-deleted context 下的 likelihood response。

判定标准：

- Llama 与 Mistral 上，counterfactual curve 都必须超过 direct full-context likelihood baseline，而不是只在单个模型有效。
- supporting-evidence dependency gap 在 correct 与 incorrect response 间方向一致。
- 若仍不稳定，CF-UQ 也立即 NO-GO，再寻找新方向。

结果：

| Backbone | Direct AUROC | CF curve AUROC | Fixed support-dependency AUROC | 判定 |
|---|---:|---:|---:|---|
| Llama-3.1-8B | **0.6689** | 0.6309 | 0.5576 | Fail |
| Mistral-7B | 0.7813 | **0.8809** | 待固定分数复核但不影响跨模型判定 | 单模型成功 |

Llama 上无论 learned curve 还是预先定义 dependency score 都低于 direct likelihood，违反跨 backbone gate，故 CF-UQ 正式 NO-GO。

### P7：Reasoning-Induced Confidence Amplification（已完成，NO-GO）

新观察来自 P4/P6 的同一样本均值：

- Llama：加入 generated reasoning 后，correct conclusion likelihood 约提升 +0.09，incorrect 约提升 +0.49。
- Mistral：correct 约提升 +0.46，incorrect 约提升 +0.76。

核心假设：错误 CoT 会为原本证据较弱的结论制造更大的后验 confidence amplification；因此 `pre-reasoning logP(answer) - post-reasoning logP(answer)` 可作为无需训练的 hallucination score。

Baseline gate：固定 amplification score 必须在 Llama/Mistral 上都超过 pre-only 与 post-only likelihood；之后再扩大样本并加入 hidden-state probe、token entropy 和 learned stack。失败则立即停止该方向。

| Backbone | Pre-only | Post-only | Fixed negative amplification | Learned pre/post |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 0.6689 | 0.5479 | **0.6982** | 0.7002 |
| Mistral-7B | **0.7813** | 0.7666 | 0.7529 | 0.7715 |

错误答案在两个模型上确实具有更大的平均 confidence amplification，但该效应不能稳定改善 ranking；Mistral 明确退化，故 P7 NO-GO。

### P8：Evidence-Bottleneck UQ（已完成，NO-GO）

动机：P6 中 `support-only` 对 correct/incorrect 的平均 log-likelihood separation 在两模型都大于 full context，但 gold supporting facts 不能作为实际推理输入。P8 从 generated reasoning 中匹配其引用/提及的 passage titles，形成无需外部 judge 的 self-cited evidence subset。

比较：

1. full-context conclusion likelihood；
2. random same-size passage subset；
3. self-cited evidence likelihood（候选方法）；
4. gold-support evidence likelihood（仅作 upper bound，不算可部署方法）。

Gate：self-cited score 或预先定义的 self-cited/full gap 必须在 Llama/Mistral 都超过 full 与 random baseline；之后才扩大样本并和 hidden-state probe、semantic/token uncertainty 比较。

Llama 结果：full=0.6689，self-cited=0.5898，reasoning-aligned=0.6348，random-aligned=0.5313，gold-title upper bound=0.7510。虽然无监督 step–passage alignment 将 gold-title recall 从 67.2% 提升至 85.9%，仍未超过 full baseline，说明失败不只是 citation recall。P8 NO-GO。

### P9：Conclusion–Reasoning CompatibilityUQ（已完成，NO-GO）

核心假设：all-good / 四象限风险可能不在线性拼接特征中，而在 conclusion 与 reasoning representation 的交互中。构造参数量匹配的显式 compatibility feature：

- normalized hidden-state elementwise product；
- normalized hidden-state absolute difference；
- cosine similarity；
- conclusion/reasoning token-confidence 及差值。

使用单个 four-way linear head，与 P5 的 direct four-way linear（33,620 params）、separate experts（16,810）和 matched scalar 比较。Llama 结果（seed=2026）：

| Method | Params | ID All-good AUROC | ID 4-way Macro-F1 | OOD mean All-good AUROC | OOD mean 4-way Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Direct four-way | 33,620 | **0.7535** | **0.3707** | **0.6929** | **0.2748** |
| Separate experts | 16,810 | 0.7408 | 0.3433 | **0.6961** | 0.2169 |
| CompatibilityUQ | 34,048 | 0.7288 | 0.3465 | 0.6696 | 0.2529 |

CompatibilityUQ 相对 separate experts 的 OOD Macro-F1 提高 +0.0360，且 OOD NLL 更低；但核心 all-good AUROC 比 direct four-way 低 0.0233、比 separate experts 低 0.0265，ID 也退化。局部分类阈值收益不足以抵消 ranking 失败，因此不在第二个 backbone 上扩展，P9 **NO-GO**。

可复现脚本：`analysis/pilot_compatibility_uq.py`；结果：`popllm/results/refactor/pilot_compatibility_uq_llama.json`。

### P10：Domain-Robust Hallucination UQ（已完成，NO-GO）

前述实验揭示的更稳定问题不是“缺少哪一种 head”，而是 HotpotQA 上训练的 uncertainty probe 在不同任务上的退化巨大。P10 将研究问题改为：**在未知 reasoning domain 上，怎样避免 hallucination detector 学到 source-specific confidence shortcut？**

小样本协议：

1. 七个数据域逐一 leave-one-domain-out；其余六个域的 `validation` cache 作为 source training，留出域的 `test` 只用于最终评价。
2. 主目标先使用 answer correctness，避免方法正结果依赖 reasoning judge；all-good 作为附加诊断。
3. 相同 frozen feature 与相同线性模型比较 token confidence、单域/多域 ERM、class-balanced ERM、domain GroupDRO、domain×failure-type GroupDRO。
4. 固定 seed=2026；主指标为跨七个 held-out domain 的 mean/worst AUROC、AUPRC、Brier 与 selective risk。
5. Go 条件：候选方法须同时超过多域 ERM 和 token confidence 的 mean/worst AUROC，不能只靠一个域或只改善阈值指标。若失败，立即停止鲁棒优化方向。

Llama-3.1-8B、七域 leave-one-domain-out 结果（seed=2026；每个 source validation 最多 1,500 样本，held-out test 最多 1,500）：

| Method | Mean AUROC | Worst AUROC | Mean Brier | Mean AURC |
|---|---:|---:|---:|---:|
| Token confidence | 0.5470 | 0.4322 | 0.3205 | 0.3764 |
| Multi-domain ERM | 0.6176 | 0.5063 | **0.2361** | 0.3353 |
| Balanced ERM | **0.6209** | 0.5263 | 0.2401 | **0.3333** |
| Domain GroupDRO | 0.6110 | 0.5233 | 0.2445 | 0.3400 |
| Domain×type GroupDRO | 0.6022 | **0.5341** | 0.2535 | 0.3472 |

type-aware GroupDRO 的 worst AUROC 相对 balanced ERM 仅提高 +0.0078，同时 mean AUROC 下降 -0.0187，Brier/AURC 也更差；不满足预先设定的 mean+worst 双重 gate。说明当前 domain/failure reweighting 没有产生更强 detector，P10 **NO-GO**。

可复现脚本：`analysis/pilot_domain_robust_uq.py`；结果：`popllm/results/refactor/pilot_domain_robust_uq_llama.json`。

### P11：Unlabeled Domain-Relative UQ（已完成，NO-GO）

假设：跨域 detector 失败来自 hidden/confidence/attention features 的域尺度漂移。对每个 source domain 使用自身 training moments 标准化；测试时只使用无标签 target batch 的 feature moments，不访问 correctness label。分类器、训练损失、参数量均与 ERM 相同。

结果：domain-relative ERM mean/worst AUROC=`0.6128/0.5158`；domain-relative balanced ERM=`0.6127/0.5252`；均未超过 global balanced ERM=`0.6209/0.5263`，Brier/ECE 还明显退化。P11 **NO-GO**。

可复现脚本：`analysis/pilot_domain_relative_uq.py`；结果：`popllm/results/refactor/pilot_domain_relative_uq_llama.json`。

### P12：Multi-Trajectory Disagreement UQ（已完成，lexical 版本 NO-GO）

研究问题切换为生成级不确定性：当多条轨迹得到相同答案时，普通 self-consistency 会给出高置信度；但若这些轨迹的中间实体/证据路径高度不一致，这种“答案共识”可能是 shortcut-driven confident hallucination。

首轮比较：

1. 单次 answer token confidence；
2. exact/normalized-answer self-consistency；
3. answer-cluster entropy（semantic entropy 的轻量可复现版本）；
4. reasoning-path lexical/semantic disagreement；
5. answer agreement 与 path disagreement 的预先定义组合及训练型 stack。

Gate：组合信号必须超过 self-consistency/answer entropy 和单次 token confidence 的 AUROC/AURC；先在 HotpotQA 小样本验证，成功后必须在第二 backbone 和第二 dataset 复现。生成协议固定 seed=2026，不运行多种子。

Llama-3.1-8B、HotpotQA 192 条、每题 8 trajectories（1,536 completions）结果：held-out self-consistency AUROC=0.7218，固定 `answer consensus × lexical path consistency`=0.7266，learned answer+path stack=0.6705。全 192 条固定分数=0.7360、self-consistency=0.7276；paired bootstrap 增益 95% CI=`[-0.0333, +0.0504]`，不能排除零效应。citation consistency 单独 AUROC=0.4910。故“path 文本相似度揭示 confident hallucination”的版本 **NO-GO**。

生成缓存：`popllm/results/refactor/multitrajectory_hotpot_llama_generations.json`；脚本：`analysis/pilot_multitrajectory_uq.py`。

### P13：Evidence-Grounded Consensus UQ（已完成，NO-GO）

与 P12 的纯 trajectory 相似度不同，P13 检查每条 modal-answer trajectory 是否被输入 context 实际支持：

1. 引用的 passage title 是否存在于输入 context；
2. reasoning step 内容词有多少可在对应 passage 中找到；
3. modal answer 是否/多频繁出现在 context；
4. `answer consensus × trajectory grounding` 固定组合，以及 answer-only / grounding-only / learned stack。

不使用 gold supporting-fact title、reasoning judge 或 target label 构造 grounding features。Gate 仍为超过 self-consistency/entropy，并要求 paired bootstrap CI 不跨 0；通过后再做 Mistral 与 2Wiki/MuSiQue。

结果：held-out self-consistency AUROC=0.7218；grounding-only=0.4764；fixed grounded consensus=0.6722；learned grounded stack=0.6213。全 192 条 fixed grounded consensus=0.6995，低于 self-consistency=0.7276。原始输出审计确认 citation parser 无误，模型实际很少按要求输出 title citation（correct/incorrect 的平均 citation validity 约 11.0%/8.7%）。P13 **NO-GO**。

脚本：`analysis/pilot_grounded_consensus_uq.py`；结果：`popllm/results/refactor/pilot_grounded_consensus_uq_llama.json`。

### P14：Self-Generated Contrastive Answer Margin（已完成，NO-GO）

多条 trajectory 不只提供 answer count，也自动产生同一问题下的 plausible alternative answers。P14 移除 reasoning prefix，在完全相同的原始 context + question 下 teacher-force 每个 unique sampled answer，计算：

- modal answer direct mean log-likelihood；
- modal 与 strongest runner-up 的 likelihood margin；
- modal 对所有 alternatives 的 normalized likelihood mass；
- consensus 与 contrastive margin 的固定组合/训练型 stack。

对比 self-consistency、answer entropy、direct likelihood、likelihood-weighted self-consistency。所有 candidates 都由模型自身采样，不使用 gold answer、supporting-fact label 或 judge。Gate：held-out 及全样本 paired bootstrap 都须显示相对最强 baseline 的清晰提升；否则 P14 停止。

结果（Llama/Hotpot，held-out 96）：self-consistency=0.7218，likelihood-weighted vote=0.7526，contrastive mass=0.7751，modal reasoning-detached direct likelihood=**0.8195**。contrastive mass 相对 weighted vote 的全样本 bootstrap 增益显著（+0.0142，95% CI `[+0.0005,+0.0305]`），但仍明显弱于 direct likelihood。受限融合 `direct + mass` 仅到 0.8223，AUPRC 反从 0.9512 降到 0.9476。故 P14 不能作为“最强方法”，**NO-GO**。

脚本：`analysis/pilot_contrastive_answer_uq.py`；结果：`popllm/results/refactor/pilot_contrastive_answer_uq_llama.json`。

### P15：Reasoning-Detached Uncertainty（已完成，NO-GO）

核心现象：模型生成 reasoning 后会把 conclusion likelihood 推高，但这种 amplification 对错误答案往往更大；与其使用不稳定的 amplification 差值，不如在 UQ 时直接切断 rationale conditioning，只保留原始 context/question 和模型已经给出的 answer。

候选 score：`log P(generated answer | original context, question, "Conclusion:")`。它不训练 head、不用 judge、不访问 gold evidence，只额外 teacher-force 一个很短的 answer。当前证据：

- P7 Llama：detached=0.6689，reasoning-conditioned=0.5479；Mistral：0.7813 vs 0.7666。
- P14 新多轨迹 Llama held-out：detached=0.8195，self-consistency=0.7218，weighted vote=0.7526，contrastive mass=0.7751。

下一 gate：完全相同的 192×8 Hotpot 协议在 Mistral 上复现；detached score 必须同时胜过 self-consistency、answer entropy、trajectory likelihood 和 contrastive/weighted vote。通过后扩第二数据集，并补 P(True)/semantic clustering baseline；失败则停止。

Mistral 结果：detached=0.6834，self-consistency=0.7199，contrastive mass=0.7248；与 Llama 上 detached=0.8195、contrastive mass=0.7751 的赢家关系相反。用 train halves 公平选择统一 rank ensemble 后，Llama AUROC 小幅改善但 AUPRC 下降，Mistral AUROC 又略低于 contrastive mass；不能得到跨模型全面占优的方法。P15 **NO-GO**。

### P16：Same-Model Verified Consensus（已完成，NO-GO）

使用与 generator 相同的 backbone，给定原始 context/question/candidate answer，teacher-force `Yes/No` 得到 context-grounded `P(True)`。对每个问题同时验证 modal answer 和 sampled alternatives，构造：

- modal `P(True)`（必须包含的强 baseline）；
- modal vs strongest alternative verifier margin / normalized verifier mass；
- self-consistency × modal verification 的固定 verified consensus；
- 受限的 answer-only 与 verifier-aware linear stack。

不使用外部 24B judge、gold support titles 或人工标签作为 verifier 输入。先在 Llama 192×8 cache 上比较 direct detached likelihood、contrastive mass、self-consistency 和 P(True)；若不能超过最强 direct baseline，P16 立即停止。

Llama held-out 结果：self-consistency=0.7218，modal P(True)=0.6309，verifier margin=0.7539，verifier mass=0.7854，learned verifier stack=0.7915；均未超过当时 raw-answer direct baseline=0.8195。canonical answer 修复进一步显示 raw direct 高分依赖答案表面形式，修正后的 Llama 最强为 self-consistency=0.7218，Mistral 最强为 likelihood-weighted vote=0.7489，但 P16 verifier 仍没有构成统一解。P16 **NO-GO**。

脚本：`analysis/pilot_verified_consensus_uq.py`；结果：`popllm/results/refactor/pilot_verified_consensus_uq_llama.json`。

### P17：Evidence-Order Invariance UQ（已完成，NO-GO）

新假设：真正由 evidence 支持的答案应对 passage 顺序近似不变；依赖首段/近因位置 shortcut 的 confident hallucination，即使在同一 prompt 重采样时保持自信，也会在证据重排后改变。

第一阶段 paired 协议（发现信号用，尚非最终 compute-matched 结论）：

1. 复用 P12 的 192 个问题与原 prompt 8 trajectories，原 modal answer correctness 是固定 target。
2. 对每题构造 8 个固定 seed 的 passage permutations，每个 permutation 采样 1 条 trajectory。permutation support 单独与 vanilla 8-sample self-consistency 同预算；固定双视角组合同时使用 ordinary 8 + permutation 8，因此总预算为 16，不能只与 vanilla-8 作最终公平比较。
3. candidate score 是 8 个 permutations 中再次产生原 modal answer 的比例；另报 permutation answer entropy 与 vanilla/permutation 固定组合。
4. 对比 vanilla self-consistency、answer entropy；对同一批 target 做 paired bootstrap AUROC delta。
5. Llama 上必须有清晰且 CI 不跨 0 的提升，才运行 Mistral；否则停止。

Llama 扩展结果（seed=2026）：

| N | Method | AUROC | AUPRC | Paired Δ AUROC 95% CI |
|---:|---|---:|---:|---:|
| 192 | vanilla self-consistency | 0.7276 | 0.8882 | — |
| 192 | fixed dual-view consensus | 0.7412 | 0.9025 | [-0.0351,+0.0679] |
| 512 | vanilla self-consistency | 0.7055 | 0.8369 | — |
| 512 | fixed dual-view consensus | 0.7272 | 0.8533 | [-0.0033,+0.0480] |
| 1,000 | vanilla self-consistency | 0.6989 | 0.8295 | — |
| 1,000 | fixed dual-view consensus | **0.7188** | **0.8479** | **[+0.0018,+0.0380]** |

Mistral 小样本跨模型结果（N=192，seed=2026）：vanilla-8 AUROC/AUPRC = 0.7199/0.8347，order support = 0.7449/0.8278，固定双视角组合 = **0.7752/0.8751**；组合相对 vanilla-8 的 paired AUROC 差值均值 +0.0824，95% CI **[+0.0272,+0.1398]**。方向与 Llama 一致。

关键解释：permutation support 单独在 Llama N=1,000 时为 0.6987，与 vanilla 0.6989 基本相同；增益来自固定几何平均把“同 prompt 重采样一致”与“证据顺序重排一致”作为两个必要条件。该组合没有训练参数，且 Llama 192→512→1,000 效应方向保持，Mistral 小样本也复现。learned invariance stack 在 held-out 上退化，因此不作为方法。

**公平性修正与最终 gate**：上述固定组合共消耗 16 次生成，而 vanilla baseline 只有 8 次，第一阶段结果只能证明“额外 order view 有互补信号”，不能证明它优于同算力强 baseline。最终实验固定前 8 条 ordinary trajectories 选出同一个 modal target，再各用 8 条独立 probe 验证：

1. `repeated-sampling baseline`：8 条额外原顺序 stochastic trajectories 对 target 的支持率；
2. `evidence-order candidate`：8 条 passage-reordered trajectories 对同一 target 的支持率；
3. 两者分别与前 8 条 self-consistency 做相同几何平均，总成本均为 16 generations，target 和标签完全相同；
4. 先在 Llama N=192 smoke；只有 evidence-order 在 AUROC、AUPRC 及 paired CI 上清楚胜过 repeated sampling，才扩到 N=1,000。

最终结果：同一 target 下，selection SC8=0.7242，额外 ordinary support=0.7373，order support=0.6916；`repeated-sampling joint`=**0.7552/0.9065**（AUROC/AUPRC），`evidence-order joint`=0.7417/0.9027，paired AUROC 差值 -0.0144，95% CI [-0.0732,+0.0474]。另外，标准 vanilla-16（按全部 16 条重新选 modal answer）为 **0.7576/0.9043**。因此 P17 未通过最关键的 compute-matched baseline gate，按预注册规则不扩样本并 **NO-GO**。

脚本：`analysis/pilot_evidence_order_uq.py`、`analysis/pilot_evidence_order_budget_gate.py`；结果：`popllm/results/refactor/pilot_evidence_order_uq_llama_{512,1000}.json`、`popllm/results/refactor/pilot_evidence_order_budget_gate_llama_192.json`。

### P18：Contextual Evidence Gain UQ（已完成，NO-GO）

P4 的 evidence deletion 在 Llama 上失败，一个关键混杂是 teacher-forced conclusion 前保留了原始生成 reasoning prefix；prefix 本身已经泄露答案，使 context 是否存在难以影响 conclusion likelihood。P18 完全移除 reasoning prefix，对同一个 canonical candidate 直接比较：

`CEG(a) = mean log P(a | question, context) - mean log P(a | question only)`。

除 modal answer 的 raw CEG 外，对每题 sampled alternative answers 做相同评分，构造 candidate-normalized evidence Bayes-factor mass，并与 self-consistency 固定组合。自然强 baseline 包括 vanilla SC、full-context direct likelihood、full-context contrastive likelihood mass、likelihood-weighted vote。第一 gate 使用现有 Llama/Mistral 192×8 Hotpot cache、固定 seed=2026；必须在两个 backbone 上方向一致并超过各自最强 baseline，否则立即停止。

Llama N=192 结果：SC=0.7276/0.8882（AUROC/AUPRC），raw modal context gain=0.5996/0.8475，candidate-normalized evidence Bayes-factor mass=0.7273/0.8926，最佳固定 SC×weighted-evidence joint=0.7280/0.8928。最佳方法相对 SC 的 paired AUROC 差值仅 +0.0014，95% CI [-0.0180,+0.0226]；没有清楚超过强 baseline。因此按 gate 不再运行 Mistral，P18 **NO-GO**。

脚本：`analysis/pilot_contextual_evidence_gain_uq.py`；结果：`popllm/results/refactor/pilot_contextual_evidence_gain_uq_llama.json`。

### P19：Set-Relative Self-Verification UQ（进行中）

重新用 canonical candidate scoring 审计 P16 后发现，原先把 raw generated answer 的表面形式错误归到 normalized answer key，导致 direct likelihood 被虚高，并遮蔽了 verifier 的真实相对优势。新假设是：LLM 对单个 candidate 的 absolute `P(True)` 受 acquiescence / prompt bias 严重影响，但同一问题内对模型自己产生的反事实 candidates 做相对归一化，可以抵消 question-level verifier bias：

`RelVerify(a*) = P(True | q,c,a*) / sum_{a in sampled candidates} P(True | q,c,a)`。

Llama N=192 canonical 复算：SC=0.7276/0.8882，full contrastive mass=0.7114/0.8893，absolute modal P(True)=0.6153/0.8627，verifier margin=0.7510/0.9055，`relative verifier mass`=**0.7583/0.9002**。held-out relative verifier mass=0.7854/0.9115，亦超过 held-out SC=0.7218 与 contrastive=0.7061。该方法不使用外部 judge、训练参数或 gold evidence。

当前 gate：先在 Mistral 相同 192×8 candidate set 上复验；若方向一致，再补与额外 stochastic sampling 的 token/FLOP-matched baseline，并扩大 N。若 Mistral 失败或同算力 repeated sampling 更强，则 P19 停止。

## 6. 方向确定后的完整实验矩阵

1. **Backbones**：Llama-3.1-8B、Mistral-7B、Gemma-2-9B、Phi-4；reasoning model（Phi-4-reasoning 或 Ministral reasoning）作为额外 setting，而非混入主表。
2. **任务族**：
   - grounded multi-hop QA：HotpotQA、2Wiki、MuSiQue、IIRC；
   - logical/process reasoning：StepGame、bAbI、StrategyQA；
   - hallucination/factuality：RAGTruth、HalluLens/TruthfulQA、false-premise 或 CHOKE setting（需核验许可与下载）。
3. **强基线**：token entropy/NLL、semantic entropy、P(True)、SAPLMA、simple factuality probe、UHead、CoT-UQ、judge-only（若任何表使用 judge）、FRANQ/相关 typed 方法（适用时）。
4. **指标**：每种 risk 的 AUROC/AUPRC/ECE/Brier；worst-group 与 cross-type transfer；risk-coverage；fixed-budget hallucination rate；反事实 selectivity。
5. **严谨性**：按用户要求固定单 seed=2026；使用 paired bootstrap CI / paired significance test、统一 feature access、统一 label access、明确 ID/OOD 切分；不再用 thresholded confidence 的 Accuracy 冒充主要 UQ 结果。
6. **标注可信度**：LLM 弱标注 + 至少 200–500 条双人/仲裁人工子集；报告 judge 与人工的一致性和按类型误差，而不是只报一个总体 agreement。

## 7. 工程重构顺序（方向确定后执行）

1. 修复 `config.py` 与 `jobs/common.sh` 的项目根目录推导，全部默认到仓库内 `popllm`。
2. 冻结旧 pipeline 到 `legacy/` 或 git tag；保留 loader/cache schema，不继续扩展旧 `chainuq_head.py`。
3. 新增 `analysis/`：P0/P1/P2 可复现实验、数据审计和统计图。
4. 新增 typed labels / counterfactual metadata schema，严禁把 verifier verdict 混入 detector feature。
5. 实现最小模型与 baseline access contract，确保每个比较使用相同信息。
6. 先跑 1 model × 2 datasets × 小样本；通过 go/no-go 后扩至 4 × 多任务矩阵。
7. 核心表稳定后重写 LaTeX：Problem → non-identifiability → method → controlled evidence → broad evaluation。

## 8. AAAI-27 时间风险与里程碑

- 2026-07-15：完成方向选择、P0/P1、锁定题目与三条贡献。
- 2026-07-16–18：完成新方法最小实现、P2 和主要 baselines；若核心假设失败，立即止损切 B。
- 2026-07-19–21：扩到 2–4 模型与 3 类任务；提交 abstract。
- 2026-07-22–25：全量主表、ablation、人工核验子集、统计显著性。
- 2026-07-26–28：完成全文、图表、limitations 与 reproducibility checklist。
- 2026-07-31：补充材料和匿名代码。

**现实判断**：从零重做并在 13 天内形成 AAAI-27 成熟长文属于高风险冲刺。若 7 月 18 日前 P2 没有清晰正结果，应停止堆实验，选择延期到下一投稿周期，而不是复用旧表强行包装。

## 9. 决策记录

- 2026-07-15：用户明确允许完全另起炉灶，judge 可有可无，但优先复用路径、脚本和 cache。
- 2026-07-15：用户确认共享资产根目录为仓库内 `/home/dy23a.fsu/popllm/ChainUQ/popllm`。
- 2026-07-15：用户要求分别运行 A/B/C 小实验后再决定；三方向 pilot 完成，当前实证推荐 A/TypeUQ，等待用户确认进入正式重构。
- 2026-07-15：用户要求后续只使用一个 seed，并持续和强 baseline 比较；TypeUQ 若不能清楚领先则换方向。固定 seed=2026，启动 P5 baseline gate。
- 2026-07-15：P5 失败，TypeUQ 未超过 direct four-way / separate experts；遵照用户要求停止 TypeUQ，切换 CF-UQ v2。
- 2026-07-15：CF-UQ v2 在 Llama 上失败，停止该方向；从 pre/post reasoning likelihood 差异中提出 reasoning-induced confidence amplification，启动 P7。
- 2026-07-15：P7 仅在 Llama 改善、Mistral 退化，停止；启动无需 gold evidence 的 self-cited evidence-bottleneck P8。
- 2026-07-15：P8 自动 evidence recall 达 85.9% 仍低于 full baseline，停止；启动显式 conclusion–reasoning interaction 的 P9。
- 2026-07-15：P9 的 compatibility interaction 未超过 direct four-way / separate experts 的 ID/OOD all-good AUROC，停止；启动 leave-one-domain-out 的 domain-robust hallucination UQ P10。
- 2026-07-15：P10 type-aware GroupDRO 只轻微改善 worst AUROC、但 mean/calibration/selective risk 均弱于 balanced ERM，停止；开始审计多信号互补性，并准备生成级 trajectory 方向。
- 2026-07-15：P11 无标签 target-relative normalization 未超过 global balanced ERM，停止；启动 B200 多轨迹生成 P12。
- 2026-07-15：P12 lexical path consistency 仅有不显著的 +0.008 AUROC，learned stack 退化，停止；从同一生成缓存转向 evidence-grounded consensus P13。
- 2026-07-15：P13 因 citation compliance 极低且 grounding/组合均弱于 self-consistency 而停止；启动 self-generated alternative answer 的 contrastive likelihood margin P14。
- 2026-07-15：P14 contrastive mass 虽超过 weighted vote，但没有超过 reasoning-detached direct likelihood；停止 contrastive 方法，改以 detached rescoring 为 P15，并启动 Mistral 复验。
- 2026-07-15：P15 detached score 在 Mistral 低于 self-consistency，统一 rank ensemble 也无法同时改善 AUROC/AUPRC；停止，启动 same-model verified consensus P16。
- 2026-07-15：P16 same-model P(True)/verifier stack 未超过强 baseline，停止；切换到同预算的 evidence-order invariance P17。
- 2026-07-15：P17 在 Llama N=1,000 上取得显著 paired AUROC +0.0197（95% CI [+0.0018,+0.0380]）且 AUPRC +0.0184；通过单模型 gate，启动 Mistral 跨模型复验。
- 2026-07-15：P17 在 Mistral N=192 复现正向结果（fixed=0.7752 vs vanilla-8=0.7199，paired 95% CI [+0.0272,+0.1398]）；同时发现 fixed 组合总预算是 16 而非 8，故将结论降为 provisional，并启动同 target、同 16-generation 预算的 repeated-sampling 强 baseline。
- 2026-07-15：P17 compute-matched gate 失败（Llama N=192：order joint 0.7417 < repeated-sampling joint 0.7552 < vanilla-16 0.7576），停止扩样本；切换到移除 reasoning leakage 的 Contextual Evidence Gain P18。
- 2026-07-15：P18 Llama gate 失败（最佳 fixed=0.7280 vs SC=0.7276，paired CI 跨 0；raw gain=0.5996），停止；转为复用现有 verifier/consensus cache 审计互补性，避免盲目新增生成。
- 2026-07-15：canonical 修正后重新审计 P16，发现 relative verifier mass 在 Llama full/held-out 均超过 SC、contrastive 与 absolute P(True)；将其独立为 P19 Set-Relative Self-Verification 并启动 Mistral gate。
- 2026-07-15：用户要求暂停；已停止实验。P19 Mistral 跨模型、compute-matched baseline 与扩样本均尚未运行，不能把 P19 表述为已验证方法。
