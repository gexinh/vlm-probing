# 知识冲突、任务组合与计数：6 篇逐篇核查

核查日期：2026-09-17。核查了官方论文全文的相关方法章节、官方仓库 README、目录和关键代码。未运行上游实验，未下载模型或数据集。6 个用户链接均有效，题名与目标工作一致；下文区分“可纳入的方法”“诊断数据/评测”“缓解策略”。P0 为首批通用能力，P1 为组合实验，P2 为可选扩展；这是本库的工程优先级，并非论文结论。

| 论文 | 身份与来源 | 建议 | 代码与许可 |
|---|---|---|---|
| HallusionBench | CVPR 2024；[论文](https://arxiv.org/pdf/2310.14566)、[官方仓库](https://github.com/tianyi-lab/HallusionBench) | P1 诊断数据适配器及成对一致性指标 | BSD-3-Clause；评测脚本可见 |
| Insight Over Sight / ConflictVIS | ACL 2025；[正式论文页](https://aclanthology.org/2025.acl-long.872/)、[全文](https://arxiv.org/html/2410.08145) | P1 冲突数据；溯源其使用的 LVLM-Interpret relevance 工具 | [ConflictVIS](https://github.com/xyliu-cs/ConflictVIS) 主要提供 README/示例/结果；未见明确仓库许可证 |
| WHOOPS-AHA! / Seeing-Knowing | ACL 2026；[正式论文页](https://aclanthology.org/2026.acl-long.642/)、[全文](https://aclanthology.org/2026.acl-long.642.pdf) | P0 head logit attribution / head reweighting；P1 冲突区域定位 recipe | [Seeing-Knowing](https://github.com/francescortu/Seeing-Knowing) 代码公开；无许可证声明 |
| Compose and Fuse | ICLR 2026；[会议页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/13192e17060e78eee6c5425740989181-Abstract-Conference.html)、[全文](https://arxiv.org/html/2509.23744v4) | P0 attention-feature linear probe / attention temperature；P1 composition recipe | [OmniReason](https://github.com/DELTA-DoubleWise/OmniReason)，Apache-2.0 |
| Visual Counting Bottleneck | ICML 2026；[全文](https://arxiv.org/pdf/2605.30170) 页 1 标注会议 | P0 linear probe / token ablation；P1 hidden-number recipe | [semproj](https://github.com/Russellpang/semproj) 代码公开；无许可证声明 |
| MME-CoT | ICML 2025；[正式论文页](https://proceedings.mlr.press/v267/jiang25n.html)、[全文](https://arxiv.org/html/2502.09621v1) | P2 输出层面的 CoT 诊断评测 | [MME-CoT](https://github.com/MME-Benchmarks/MME-CoT) 评测代码公开；无许可证声明 |

“无许可证声明”指本次检查未发现仓库级明确授权；不将公开可读等同于可直接复制进本库。可先登记论文方法、链接及独立接口设计，代码复用再按对应授权处理。

## 1. HallusionBench

论文构造原图/编辑图/无图对照，并按 question、figure、question pair 计算正确率。它适合作为检查模型是否响应视觉证据变化的行为诊断，也能提供 activation patching 的候选 clean/corrupt 配对；后者是本库建议，原论文没有提出统一的内部激活 patching 算法。[论文 §3–4](https://arxiv.org/pdf/2310.14566)

接入建议：`HallusionBenchDataset` 保留 `set_id/figure_id/question_id/visual_input`，`PairConsistency` 在组内对齐后评分。候选 patching 配对还须验证题目、答案变化以及视觉 token 对齐，不能自动把所有 question pairs 当作可 patch 的等长输入。[数据与字段说明](https://github.com/tianyi-lab/HallusionBench)

现有入口：[evaluation.py](https://github.com/tianyi-lab/HallusionBench/blob/744007c232c292942c7f80eb61edb2465482da31/evaluation.py)、[utils.py](https://github.com/tianyi-lab/HallusionBench/blob/744007c232c292942c7f80eb61edb2465482da31/utils.py)。许可为 [BSD-3-Clause](https://github.com/tianyi-lab/HallusionBench/blob/744007c232c292942c7f80eb61edb2465482da31/LICENSE.md)。不要照抄 README 的历史下载统计作为当前规模，加载具体数据版本后计算记录数。

## 2. Insight Over Sight / ConflictVIS

有适合本库的分析，但需要正确归属：§4.4/Fig.7 使用 Stan et al. 的输入到输出 relevance 分数检查视觉信息利用，所引工作是 **LVLM-Interpret**。FoV 是 prompting 缓解策略；VCD、PAI 和 CoT/SFT 是被评估的既有干预/训练方案，不能登记成该论文新提出的 probing primitive。[方法全文](https://arxiv.org/html/2410.08145)

官方 [ConflictVIS 仓库](https://github.com/xyliu-cs/ConflictVIS) 给出 `xiaoyuanliu/conflict_vis` 数据加载和推理示例，当前根目录仅见 README、assets、results，未见独立 attribution 实现。因此纳入 `ConflictVisDataset`，relevance 方法溯源至 [IntelLabs/lvlm-interpret](https://github.com/IntelLabs/lvlm-interpret)。后者 Apache-2.0，提供 `utils_attn.py`、`utils_relevancy.py` 和 causal-discovery 工具；论文为 [CVPR 2024 XAI4CV workshop](https://openaccess.thecvf.com/content/CVPR2024W/XAI4CV/papers/Stan_LVLM-Intrepret_An_Interpretability_Tool_for_Large_Vision-Language_Models_CVPRW_2024_paper.pdf)。建议拆出 attention/relevance 计算，Gradio UI 不进入基础依赖。

## 3. WHOOPS-AHA! / Seeing-Knowing

这是本组最直接的 mechanistic probing 来源。§4.2 将 residual/attention/MLP/head 输出投影至词表，比较知识一致与图像一致候选 token，识别方向相反的 attention heads。随后在最后一个 query 位置，对特定 heads 的 image/text attention 权重做 post-softmax 乘法干预。还提供高 attention visual-token 选择、梯度归因对照及分割区域内/外 attribution ratio。[论文 §4.2–5.4](https://aclanthology.org/2026.acl-long.642.pdf)

建议用已有 primitives 组合成 `KnowledgeConflict` recipe：`ComponentLogitLens` → 候选 token logit difference / head ranking → `AttentionReweight` → `VisualAttribution`。必须保留 pre/post-softmax、是否重新归一化、query 位置和 key modality；reweight、pre-softmax knockout、activation replacement 不是同一种操作。单个 head 结果须经过该 head 对应的 output projection 后才处于 residual 空间；不能直接将 head_dim 张量送入 LM head。上述接口约束是本库设计要求。

官方运行入口为 [script/1_logitlens.py、2_intervention.py、3_pixel_localization.py](https://github.com/francescortu/Seeing-Knowing/tree/830f6752e1c2af0120242eb3b8120120b20166ef/script)。关键复用参照：[ablation_utils.py](https://github.com/francescortu/Seeing-Knowing/blob/830f6752e1c2af0120242eb3b8120120b20166ef/script/experiment/1_heads_ablation/ablation_utils.py)、[visual_attn.py](https://github.com/francescortu/Seeing-Knowing/blob/830f6752e1c2af0120242eb3b8120120b20166ef/src/visual_attn.py)。仓库基于 `easyroutine` 的 `VisualComp` 分支和 Transformers 4.51.1；论文/README 支持 LLaVA-NeXT-7B 与 Gemma3-12B。两类视觉切块映射由不同类实现，不能假设通用方形网格。[依赖](https://github.com/francescortu/Seeing-Knowing/blob/830f6752e1c2af0120242eb3b8120120b20166ef/pyproject.toml)

## 4. Compose and Fuse

§4 中有三项明确可泛化能力：对生成 token 到各 fact 的 attention 做汇总，训练 linear probe 分别预测 fact usefulness 与 modality；由 layer/head probe 系数分析信息分布；对指定层 attention temperature 做干预。论文另用“识别后推理”的两步 prompting 作行为对照。[全文 §4 与附录 A.2](https://arxiv.org/html/2509.23744v4)

官方已提供 [extract_attention.py](https://github.com/DELTA-DoubleWise/OmniReason/blob/7a5a1ef471856b2fe4c093904c1981e7ffa42781/src/interpretation/extract_attention.py)、[probe_linear_layerwise.py](https://github.com/DELTA-DoubleWise/OmniReason/blob/7a5a1ef471856b2fe4c093904c1981e7ffa42781/src/interpretation/probe_linear_layerwise.py)、[attention_manipulation.py](https://github.com/DELTA-DoubleWise/OmniReason/blob/7a5a1ef471856b2fe4c093904c1981e7ffa42781/src/interpretation/attention_manipulation.py)。代码采用 `GroupKFold` 按 sample id 分组；temperature 实现通过 q projection hook 对每个 head 的 query 除以 T，直接脚本主要面向 Qwen2.5-Omni。[Apache-2.0](https://github.com/DELTA-DoubleWise/OmniReason/blob/7a5a1ef471856b2fe4c093904c1981e7ffa42781/LICENSE)

建议复用 `LinearProbe` 通用类，输入可为 hidden states 或 attention-derived features；`AttentionTemperature` 独立于 decoder sampling temperature。fact spans、source modality 与 sample-group split 必须在数据层显式提供。probe 准确率只支持可解码性，系数大小不单独证明因果重要性；因果结论由后续 intervention 和对照支持。

## 5. Visual Counting Bottleneck

论文以可控棋盘分离感知、大小比较与符号输出。可抽取 patch-level 黑子分类并累计得到 hidden number、跨层 linear probes、视觉/文本 attention-head deactivation 对照，以及 probe 选出的视觉 tokens 的 attention masking。后者检验去掉 k 个目标后输出是否相应减 k。[全文 §2.3、§3、§4、附录 D](https://arxiv.org/pdf/2605.30170)

代码入口：[hidden_number_probing_clf.py](https://github.com/Russellpang/semproj/blob/8fb587281f815e2ed6cf2493707c3013d9559ae3/hidden_number_probing_clf.py)、[causal_hidden_number_intervention.py](https://github.com/Russellpang/semproj/blob/8fb587281f815e2ed6cf2493707c3013d9559ae3/causal_hidden_number_intervention.py)、[test_head_deactivation_vision.py](https://github.com/Russellpang/semproj/blob/8fb587281f815e2ed6cf2493707c3013d9559ae3/test_head_deactivation_vision.py)。Qwen3-VL 另有 [vision probe](https://github.com/Russellpang/semproj/blob/8fb587281f815e2ed6cf2493707c3013d9559ae3/train_hidden_num_qwen3vl.py) 与 [LM layer probe](https://github.com/Russellpang/semproj/blob/8fb587281f815e2ed6cf2493707c3013d9559ae3/train_lm_layer_hidden_num_qwen3vl.py)，使用 PCA 加 logistic regression/linear SVM。

适合构成 `HiddenNumber` recipe，核心只需 `LinearProbe`、`TokenAblation`、`HeadAblation`；棋盘生成、标签到 patch 对齐、计数聚合和 ID/OOD 划分属于 recipe。不能将棋盘上的高 probe 准确率推广为自然图像计数机制已完全解释。论文自定义模型与 Qwen3-VL 验证的数据切分不同，适配器需分别记录。[仓库 README](https://github.com/Russellpang/semproj)

## 6. MME-CoT

这项工作评估可见生成过程。质量对应 step precision/recall，鲁棒性比较 direct 与 CoT prompting 的 stability/efficacy，效率包含 relevance rate/reflection quality；没有引入激活级因果追踪方法。[全文 §2](https://arxiv.org/html/2502.09621v1)

建议放入 `evaluation` 插件：`CoTDiagnostics` 接收生成文本、reference steps 和 direct/CoT 配对结果；不作为 `Probe` 的子类。官方入口 [main.py](https://github.com/MME-Benchmarks/MME-CoT/blob/08fa4e9144d47a69d5129376b8f3f449c28a2298/main.py)、[direct_eval.py](https://github.com/MME-Benchmarks/MME-CoT/blob/08fa4e9144d47a69d5129376b8f3f449c28a2298/direct_eval.py)、[final_score](https://github.com/MME-Benchmarks/MME-CoT/tree/08fa4e9144d47a69d5129376b8f3f449c28a2298/final_score)。部分评测依赖 GPT judge，需保存 judge/model/prompt 版本和缓存；仅启用用户所选评测，不能把外部 judge 调用作为基础 probing 的默认步骤。[运行说明](https://github.com/MME-Benchmarks/MME-CoT)

## 对统一 library 的具体约束

以下为基于上述代码和方法的设计建议，尚非已实现 API：

1. `LinearProbe.fit/evaluate` 统一 hidden-state、patch、attention-feature 三种输入，使用 `FeatureSet(values, targets, group_ids, coordinates)`；split 在标准化/PCA 之前，所有拟合仅用训练 fold。
2. `AttentionReweight`、`AttentionTemperature`、`AttentionKnockout` 分别定义干预位置及数学操作，通过 model adapter 获取结构化 layer/head/query/key 选择，禁止用同一个含糊的 `scale` 混合表示。
3. `ComponentLogitLens` 输出 component-level vocab scores；候选 token 对比、head ranking、冲突数据过滤由 recipe 做，避免每篇论文都复制一套模型加载与 hook 逻辑。
4. `VisualAttribution` 只返回 token scores；坐标回映射委托 `VisualTokenMap` 处理 resize、crop、merge、padding。ROI 内/外强度比和 mask overlap 都作为显式指标，零背景分母要单独标记。
5. `TokenAblation` 与 `HeadAblation` 给出干预前后配对指标，并保留随机选择、同数量非目标选择等对照。`HiddenNumber` 和 `KnowledgeConflict` 只组合 primitives 与任务指标。
6. benchmark adapters 和 CoT judges 独立于 model hooks；不会把数据集、训练方案、提示策略强行建成探针类。

本次源码检查为静态检查，不能据此声称兼容当前 Transformers 或已复现论文数值。库初期应先验证一类 decoder-only VLM，随后逐 adapter 增加能力。
