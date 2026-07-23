# ChainUQ：面向 KDD 稳健投稿的新增实验路线图

> 目标不是承诺“稳收”，而是系统性消除最可能导致拒稿的解释空间。KDD Research Track 明确考察 originality、technical merit、potential impact、quality of execution 和 reproducibility，且投稿前 8 页必须自洽。官方要求见 [KDD 2027 Research Track CFP](https://kdd2027.kdd.org/research-track-call-for-papers/)。

## 1. 当前最危险的 reviewer interpretation

现有 31-cell 数学实验已经证明 ChainUQ 在域内 correctness ranking 上有效，但仍可能被解释为：

1. **“只是每个 model-dataset cell 单独训练的监督特征工程。”** 当前 logistic head 是 within-cell verifier-supervised，尚未证明跨数据集、跨模型或统一 head 的迁移能力。
2. **“只是收集了多个 prefix 的静态置信度，而不是真正利用 temporal dynamics。”** Endpoint-controlled ablation 已证明不是最后一个点，但还没有严格证明时间顺序本身不可替代。
3. **“只对数学题有效。”** 当前五个 benchmark 都是可验证数学任务，无法支持一般 reasoning UQ 的表述。
4. **“强 baseline 不够完整。”** 需要在完全相同的 cache、标签、fold 和成本合同下加入更强的 prefix、hidden-state、verbalized confidence 和 sampling UQ 对照。
5. **“低成本只是 token-count proxy。”** 目前只有 decoded-token overhead，没有 wall-clock、吞吐、峰值显存和 prefix-cache 消融。
6. **“统计显著性可能受 cell 内重复拟合、multiple comparisons 或小型 AMC23 影响。”** 需要预先固定 primary contrasts，并提供跨 cell 的总体检验和稳健性分析。

因此，新增实验必须围绕三条核心问题展开：

> **Does the signal transfer? Does temporal order matter? Does it survive outside mathematics?**

---

## 2. P0：投稿前必须完成的实验

### P0-0. 统一实验口径与结果溯源（release blocker）

这不是新模型实验，但必须先完成。`docs/EXPERIMENTS.md`、旧日志和当前 `master_table.json` 中存在不同阶段的数值口径。任何新增实验都只能接入一套冻结协议，否则 reviewer 一旦发现表格间数值无法复现，其他实验没有意义。

**冻结内容：**

- 唯一 cell manifest：dataset、model、样本数、缺失 cell 及缺失原因。
- 唯一 generation cache、probe cache、answer normalization 和 correctness label。
- 固定 problem-level fold ID；所有方法复用完全相同的 fold。
- 标准化器、特征选择和 classifier hyperparameter 只在 training fold 拟合。
- 明确区分 primary trace、SC samples 和额外模型调用。
- 每个 JSON 写入 git commit、cache hash、feature version、seed、fold、模型版本和运行时间。
- 将 legacy 结果移入单独 namespace，禁止复制到新表。

**建议产物：**

- `$EXP_ROOT/manifests/canonical_v2.json`
- `$EXP_ROOT/results/canonical_main_v2.json`
- `$EXP_ROOT/results/canonical_ablation_v2.json`
- `experiments/audit_protocol.py`

**完成判据：** Overall、ablation、appendix matrix 能由一个命令从 canonical cache 重建，且论文中同一方法同一 cell 的值完全一致。

### P0-1. Cross-dataset、cross-model 与 unified-head transfer

这是最重要的新实验，直接回应“每个 cell 单独拟合”的攻击。

#### P0-1A. Leave-one-dataset-out（LODO）

- 对六个具有完整五数据集矩阵的 backbone，使用四个数学数据集训练一个 head，在完全未见的第五个数据集测试；只有 MATH500 的 Llama 不进入 LODO aggregate。
- AMC23 样本过小，只作为 test domain，不应主导训练或总体结论。
- 所有 feature normalization 仅使用四个 source datasets。
- 不允许把 dataset ID、平均准确率或 test-domain statistics 输入 head。

#### P0-1B. Leave-one-model-family-out（LOMO）

- 在完整 family 之间执行 leave-one-family-out；不完整的 Llama 只作为额外 MATH500 target，不参与主 aggregate。
- 同时报告同系列 size transfer，例如 Qwen3-4B/8B 训练、Qwen3-14B 测试，以及跨架构 transfer，例如 Qwen → Ministral/Phi/Llama。
- 对 tokenizer-sensitive confidence 特征，训练端统一做 robust scaling；不能使用目标模型标签重新校准。

#### P0-1C. Global unified head

- 在所有 training cells 上训练一个共享 ChainUQ head，然后对 held-out problems 做预测。
- 与当前 per-cell head 并列，而不是替换当前结果。
- 增加 `global-with-cell-ID` 仅作为 upper bound；主要结果必须是不含 cell identity 的 global head。

**Baselines：** mean log-probability、self-certainty、P(True)、answer convergence、endpoint confidence、CONV+FINAL、sequence-only logistic head。学习型 baseline 必须使用同一 source/target split。

**Primary metrics：** macro AUROC、AURC、每个 held-out domain 的 paired bootstrap CI。额外报告 worst-domain AUROC 和 source→target degradation。

**Primary contrast：** `ChainUQ_transfer - max(CONV+FINAL_transfer, sequence-only_transfer)`。

**Go/No-go：**

- 理想：LODO 和 LOMO 的 macro gain 均为正，95% CI 不跨 0，且大多数 held-out domains 为正。
- 可接受：LODO 稳定为正、LOMO mixed；论文收缩为 cross-dataset transfer，不声称 universal cross-model calibration。
- 若 transfer 接近随机或低于 endpoint baseline：必须把方法定位为 per-domain calibrated UQ，并弱化一般性。这会显著降低 KDD 竞争力。

**建议脚本与输出：** `experiments/transfer.py`，输出 `transfer_lodo.json`、`transfer_lomo.json`、`transfer_global.json`。

### P0-2. 非数学 reasoning generalization

至少加入一个足够大的非数学任务族，最好包含多个 exact-match 或 multiple-choice benchmark，以保持 judge-free correctness label。

**推荐组合：**

- BBH logical deduction / tracking shuffled objects / date understanding 等需要显式 reasoning 的任务。
- MMLU-Pro reasoning-heavy subsets，或其他具有确定选项标签的高难度多选任务。
- GPQA 可作为高难 science reasoning 补充，但样本较小，不应单独承担 generalization claim。
- 如果资源允许，可加入 HumanEval/MBPP，以 unit tests 作为 verifier；但代码答案的 prefix equivalence 与第一 token confidence 需要单独定义，优先级低于 multiple-choice。

**最低配置：** 3 个非数学任务 × 3 个不同模型 family；每个任务应有足够的正确和错误样本以稳定估计 AUROC。

**实验分两层：**

1. `in-domain non-math`：证明 ChainUQ 不依赖数学 answer parser。
2. `math → non-math transfer`：只用数学训练 head，直接测试非数学任务。这是更强的 novelty/generalization 证据。

**Baselines：** 与 P0-1 相同，额外包含 SC@2/4/8 和 semantic entropy（若语义聚类合同明确）。

**Primary contrast：** 非数学任务上 `ChainUQ - CONV+FINAL`；transfer 设置单独报告。

**Go/No-go：** 至少 2/3 任务族上超过最强 single-trace baseline，且 pooled CI 为正。若只在数学有效，标题、abstract 和 conclusion 必须显式限定 mathematical reasoning。

**建议脚本与输出：** `experiments/non_math_generalization.py`，输出 `non_math_indomain.json` 和 `math_to_nonmath.json`。

### P0-3. Temporal-order ablation：证明 novelty 真的是“trajectory”

现有 ablation 证明 pre-final probes 有用，但还不能区分“时间轨迹”与“多个 probe 值构成的无序集合”。必须加入保持 marginal information、只破坏顺序的控制实验。

#### 核心控制

1. **Endpoint only**：只保留最终 confidence。
2. **CONV only**：只保留 answer convergence。
3. **CONV + FINAL**：现有 null hypothesis。
4. **Bag-of-probes**：保留所有 probe confidence 的 mean/min/max/std/quantile，但删除 slope、early/late、commit time 和位置编码。
5. **Random permutation**：对每条轨迹随机打乱 probe 顺序，保留完全相同的 confidence multiset 和 identity counts。
6. **Reverse time**：反转 probe 顺序，测试方向信息是否必要。
7. **Ordered confidence trajectory**：仅 confidence dynamics。
8. **Ordered identity trajectory**：仅 identity dynamics。
9. **Dual ordered trajectory**：identity + confidence。
10. **Full ChainUQ**：dual trajectory + sequence features。

**重要协议：** permutation control 应同时提供两种版本：

- `train original → test permuted` 用于衡量已训练模型对顺序破坏的敏感性，但它包含 distribution shift。
- `train permuted → test permuted` 才是主要对照，因为它保持 train/test 一致，能检验无序信息本身是否足够。至少使用 10 个 permutation seeds。

**Primary contrasts：**

- `ordered trajectory - bag-of-probes`
- `ordered trajectory - permuted-trained`
- `full - CONV+FINAL`

**Go/No-go：** Ordered 必须稳定超过 bag/permuted。如果差异接近 0，论文不应继续强调“temporal dynamics”；更准确的名称应改为 multi-prefix commitment evidence。

**建议脚本与输出：** `experiments/ablation_temporal_order.py`，输出 `ablation_order.json`。

### P0-4. 强 baseline 补全与成本合同

在开始运行前先做一次最新文献搜索，逐一确认 closest work 的任务、输入、监督、模型访问级别和计算成本。所有 baseline 必须被分到可比较的 cost/access block，不能把 hidden-state、额外 forward 和多样本方法混在同一排名中。

**Single-trace / token-probability baselines：**

- Mean log-probability、token entropy、self-certainty、DeepConf variants。
- P(True) 与 verbalized confidence；明确其额外 forward/pass 成本。
- Answer convergence 与最接近的 prefix-answer/probing 方法。
- Prefix consistency / resampled continuation baseline；即使成本高，也可作为 nearest-method reference。

**Representation baselines：**

- Last-token/final-layer hidden-state linear probe。
- Mean-pooled reasoning hidden-state probe。
- 若资源允许，加入一个公开 outcome reward/verifier score；必须说明它需要额外模型，不能放入 same-cost block。

**Sampling baselines：**

- SC@2/4/8、semantic entropy@2/4/8、confidence-weighted voting。
- ChainUQ+SC 必须与相同 full-sample 数量的 SC 比较。

**Classifier-capacity control：** 对完全相同的 ChainUQ features 比较 logistic regression、XGBoost/Random Forest、MLP，以及一个直接读取序列的 GRU/TCN。主要目的不是追求最高分，而是证明增益来自 observation，而不是某个 classifier 容量优势。

**Primary table 要求：** access、full generations、extra forward、probe decoded tokens、是否需要 hidden states、是否需要训练标签必须单独列出。

### P0-5. 真实系统效率实验

当前 1.4%--3.8% 只是 decoded-token overhead，不能直接写成 latency overhead 或约等于 1× cost。

**测量对象：** ChainUQ M=2/4/8、P(True)、SC@2/4/8、无 prefix cache 的 ChainUQ、仅生成 primary trace。

**固定环境：** 至少 Qwen3-8B；最好增加一个 14B 模型。固定 GPU、CUDA、vLLM、dtype、tensor parallel、batch size、prompt/trace length bins。

**必须报告：**

- end-to-end latency（median、p90、p95）
- throughput（queries/s）
- time to first token 与 probe decoding time
- peak GPU memory
- decoded/prefill tokens
- number of full generations、probe calls 和 extra forwards
- prefix caching on/off

**展示方式：** AUROC-latency Pareto curve，不只给单个平均数。按 trace-length tercile 分层，防止长短样本混合掩盖成本。

**Go/No-go：** 只根据实测结果写“low overhead”。如果 latency 明显高于 token ratio，论文必须解释调度与多次 probe call 的成本，不能继续用 `1.0×`。

**建议脚本与输出：** `experiments/system_efficiency.py`，输出原始 per-query `system_efficiency.jsonl` 和聚合 `system_efficiency_summary.json`。

### P0-6. 统计协议升级

**预先固定三个 primary contrasts：**

1. ChainUQ vs strongest comparable single-trace baseline。
2. Ordered trajectory vs CONV+FINAL / bag-of-probes。
3. ChainUQ+SC@8 vs SC@8。

**必须增加：**

- 每个 primary contrast 的 cell-level paired bootstrap CI。
- 跨 cell 的 hierarchical bootstrap，cell 等权，避免 GSM8K 大样本主导。
- 对 31 个 cell 的 secondary tests 做 Holm correction；同时保留未校正结果但明确标注。
- 报告去掉 AMC23 后的 macro 结果。
- 报告 micro、macro、worst-dataset；主结论以 macro 为准。
- 对学习型方法使用 nested CV 或固定预注册 hyperparameter；不能在 test fold 上选 C、feature group 或阈值。
- 对核心新实验至少 3 个 generation seeds；廉价 cached ablation 使用 5--10 个 split/permutation seeds。

**不要使用：** 只对 fixed out-of-fold score bootstrap 然后暗示包含 retraining uncertainty。若不重训，caption 必须明确检验范围。

---

## 3. 建议进入正文的大 Ablation Study

大表必须围绕 reviewer 的替代解释组织，而不是把所有 feature combination 全排列。建议主表包含两个 block，完整 per-cell 和敏感性放 appendix。

### Block A：Novelty isolation

| Row | Feature configuration | 控制的替代解释 | 必须报告 |
|---|---|---|---|
| A1 | FINAL confidence only | 只是最终置信度 | AUROC/AURC |
| A2 | Answer convergence only | 只是 answer identity stability | AUROC/AURC |
| A3 | CONV + FINAL | 最强 endpoint null | AUROC/AURC |
| A4 | Bag-of-probes | 多点静态集合已足够 | AUROC/AURC |
| A5 | Permuted trajectory, retrained | 顺序无关 | mean±std over permutations |
| A6 | Ordered confidence trajectory, no endpoint | pre-final confidence dynamics | AUROC/AURC |
| A7 | Ordered dual trajectory, no sequence features | identity/confidence complementarity | AUROC/AURC |
| A8 | Full ChainUQ | 完整方法 | AUROC/AURC、Δ、CI |

### Block B：Named-component decomposition

| Row | Configuration | 解释 |
|---|---|---|
| B1 | Sequence features only | 不 probing 的静态 base |
| B2 | PrefixProbe + identity only | 只有 identity trajectory |
| B3 | PrefixProbe + confidence only | 只有 confidence trajectory |
| B4 | PrefixProbe + DualTrace | 两条轨迹是否互补 |
| B5 | PrefixProbe + DualTrace + sequence features | 单轨迹完整 ChainUQ |
| B6 | B5 + ConsensusFusion@8 | matched-budget full system |
| B7 | SC@8 only | B6 的严格 matched-budget null |

### 必须放 appendix 的细粒度 ablation

- Probe 数量 `M ∈ {1,2,4,6,8,12}`。
- Cut strategy：uniform-by-step、uniform-by-token、random cuts、first-half only、second-half only。
- Probe cue：至少 3 个语义等价 cue；报告 mean 和 worst-cue。
- Max answer tokens：8/16/32。
- Confidence definition：first-token log-prob、mean answer-token log-prob、minimum answer-token log-prob、top-2 margin（若可获取）。
- Identity equivalence：exact normalized、symbolic parser、multiple-choice exact match。
- Leave-one-statistic-group-out：level、volatility、slope/area、commitment timing、sequence features。
- Classifier：logistic、tree ensemble、MLP、temporal model。
- Label noise：随机翻转 5%/10% correctness labels，检验 verifier error sensitivity。

### Ablation 的解释规则

- 不要求每个 named component 在每个 cell 都提升；要求 primary aggregate contrast 有稳定正增益。
- 若 confidence-only 已等于 full，删掉不必要的 identity component，不要为了 framework 好看保留模块。
- 若 ordered 与 bag/permuted 无差异，必须修改 novelty claim，不能只把失败结果藏到 appendix。
- 若 ConsensusFusion 只在部分数据集有效，明确定位为 optional higher-budget extension。

---

## 4. P1：高价值 novelty amplifier

### P1-1. Self-consistency blind-spot stress test

已有 high-agreement subset 结果非常适合突出 novelty，但需要升级为预定义 stress test，而不是事后挑选阈值。

- 阈值固定为 vote fraction ≥ 0.625、0.75、0.875、1.0。
- 同时按 SC entropy quantile 分层，避免只看一个阈值。
- 在 accuracy-matched 或 difficulty-matched subsets 上比较，排除“只是容易题”的解释。
- 报告 subset size、accuracy、错误数、ChainUQ AUROC/AURC 和 fusion gain。
- Primary statement：在 SC score 近似常数的高共识区域，ChainUQ 是否仍能排序 confident-consensus errors。

这项结果应进入正文，最好替换当前较弱的 family-summary 表或其中一张描述性轨迹图。

### P1-2. Within-problem matched-pair analysis

已有 `within_problem.py` 结果应扩展并进入论文：对同一问题的正确和错误 traces 配对，问题难度、题面、gold answer 和 domain 完全相同。

- 每个问题至少有一个正确和一个错误 sample。
- 比较 ChainUQ、log-probability、P(True)、answer convergence 和 SC-derived sample score。
- 使用 paired accuracy：随机抽一对 correct/wrong trace，置信度排序正确的概率。
- 按 answer-switch count、trace length 和 final consensus 分层。

这是比简单 length control 更强的 confound control，可直接支持“commitment signal is not merely problem difficulty”。

### P1-3. Label-efficiency 与 calibration-set scaling

Verifier-supervised 方法会被问需要多少标签。对每个 training cell 使用 1%、2%、5%、10%、25%、50%、100% 标签训练，保持 test fold 不变。

- 横轴同时报告 labeled problems 数量和比例。
- 比较 ChainUQ、sequence-only、CONV+FINAL、hidden-state probe。
- 画 AUROC vs labeled examples，并报告达到 full-data 95% 性能所需样本数。
- 增加跨 cell pooled head，可证明多域数据能否减少目标域标签需求。

### P1-4. Probe robustness 与 prompt invariance

- 使用 3--5 个不改变语义的 answer cues。
- 更换 step segmentation：paragraph、sentence、token-uniform。
- 改变 thinking-end marker；对不同 model template 分开处理。
- 报告 mean、std 和 worst-case，而不是只挑最佳 cue。
- Cue 不允许用 test label 选择。

若结果对 cue 很敏感，应将 prompt ensemble 或 cue calibration 作为方法的一部分，并在成本表中计入。

### P1-5. Behavioral intervention / recovery cases

用于增强解释性，不作为因果证明：

- 将 traces 按 early-wrong→final-correct、early-correct→final-wrong、stable-correct、stable-wrong 分类。
- 对每类报告 identity trajectory、confidence trajectory、ChainUQ score 和 error rate。
- 在同一问题内展示 3--5 个 paired cases。
- 可进行 prefix truncation：只观察前 25%/50%/75% 轨迹，检验 UQ signal 何时出现。

禁止把该实验写成“visible CoT causally generates the answer”；它只能支持 behavioral diagnostic interpretation。

---

## 5. P2：资源允许时再做

1. **Code reasoning with unit-test verifier**：价值高，但 answer parsing 和 confidence definition 需要重新设计。
2. **Cross-version transfer**：例如同 family base/instruct/reasoning variants，检验 post-training shift。
3. **Calibration under distribution shift**：ECE/Brier、global temperature scaling、coverage at fixed risk。
4. **Conformal risk control**：在 calibration set 上选择阈值，评估 unseen domain 的 risk guarantee；适合 KDD，但会扩展论文主线。
5. **Adaptive compute policy**：基于 ChainUQ 决定是否追加 SC samples；只有显著改善 AUROC-cost 或 risk-cost frontier 才进入正文。现有 accuracy routing 为负，不应重复包装。
6. **Very-large-model validation**：至少一个 30B/70B 或闭源 reasoning model；只有能稳定获取 token log-prob 才适用。
7. **Human-facing selective review simulation**：将 abstained cases 送入 stronger verifier/LLM，评估错误发现率与成本。

---

## 6. 推荐执行顺序

### Stage 0：先消除口径风险

1. P0-0 canonical protocol。
2. 重建 current overall 与 ablation，冻结 primary contrasts。
3. 确认所有 cached features 可在相同 fold 上重用。

### Stage 1：优先跑不需要新生成的高信息实验

1. P0-3 temporal-order / bag-of-probes ablation。
2. P0-1 unified head、LODO、LOMO。
3. P0-6 hierarchical statistics 与 Holm correction。
4. P1-1 high-consensus stress test。
5. P1-2 within-problem matched pairs。
6. P1-3 label-efficiency。

这些实验大部分可以从现有 cache 重算，信息增益最高。

### Stage 2：需要新 generation/probing 的实验

1. P0-2 非数学任务。
2. P0-5 系统效率。
3. P1-4 cue、segmentation 与 probe-count robustness。
4. 补齐最强 baseline 所需的新输出。

### Stage 3：论文收口

1. 根据真实结果决定保留“temporal dynamics”还是改为“multi-prefix commitment”。
2. 用 transfer/non-math 结果替换正文中信息密度较低的 family 表或 synthesis prose。
3. 主文只保留 overall、大 ablation、transfer/generalization、mechanism/blind-spot 和 Pareto 结果。
4. 完整 matrix、所有 cue/seed/statistics 放 appendix。
5. 发布匿名 artifact，包含一键重建表格和 figures 的脚本。

---

## 7. “冲稳 KDD”的最小实验包

如果时间有限，至少完成以下七项：

- [ ] Canonical protocol 与所有现有数值统一。
- [ ] LODO + LOMO + global head transfer。
- [ ] 至少 3 个非数学 reasoning tasks。
- [ ] Ordered vs bag-of-probes vs permuted trajectory 的决定性 ablation。
- [ ] 最近且最接近的 prefix/trajectory UQ baseline，以及 hidden-state linear probe。
- [ ] 真实 latency/throughput/memory/Pareto 实验。
- [ ] Hierarchical CI、Holm correction、remove-AMC23 和 multi-seed audit。

理想增强项：

- [ ] High-consensus blind-spot stress test 进入正文。
- [ ] Within-problem matched-pair experiment 进入正文。
- [ ] Label-efficiency curve。
- [ ] Cue/segmentation robustness。

完成前七项后，ChainUQ 的主张才可以从：

> “在 31 个数学 model-dataset cells 上有效的域内 UQ estimator”

提升为：

> “一种可迁移、跨任务、顺序敏感的 within-trace commitment signal；它在低预算下改善 correctness ranking，并在相同采样预算下覆盖 self-consistency 的高共识错误盲区。”

---

## 8. 预期正文布局

8 页正文中优先保留：

1. **Overall table**：single-trace、extra-forward、sampling 三个成本 block。
2. **Large ablation table**：endpoint、bag、permutation、ordered、dual、full。
3. **Transfer/generalization table**：LODO、LOMO、math→non-math。
4. **One mechanism figure**：high-consensus blind spot 或 within-problem matched result；比纯均值轨迹更有判别力。
5. **One efficiency plot/table**：AUROC-latency Pareto。

建议移到 appendix：完整 family matrix、12-cell length table、所有 cue/sensitivity、完整 fusion matrix、全部 seed 结果。正文 family-summary 表的价值低于 transfer/non-math 表。

## 9. 结果解释纪律

- 不把 AUROC ranking 写成 probability calibration。
- 不把 decoded-token ratio 写成 latency ratio。
- 不把 within-cell CV 写成 domain transfer。
- 不把 permutation-test 失败隐藏起来。
- 不声称 single-trace ChainUQ 普遍超过 SC@8。
- 不根据 test performance 选择 cue、feature group 或 dataset subset。
- 对所有 negative cells、negative transfer 和 failed baselines 保留完整记录。
- 新结果若与旧文档冲突，以 canonical manifest 重算结果为准，并明确废弃旧版本。
