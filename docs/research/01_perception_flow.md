# 感知表征与跨模态信息流：8 篇逐篇核查

核查日期：2026-09-17。检查了论文正文、作者项目页，以及能定位的官方 GitHub 当前源码；没有下载模型或运行论文实验。机器可读记录与固定 revision 见 [papers.json](../../catalog/papers.json)。P0 表示首批核心原语，P1 表示后续分析/组合方法，P2 表示可选实验配方；这些是本 library 的设计建议，不是论文排名。

## 收录决策

| 工作 | 实际可复用内容 | 建议位置 | 当前代码状态 |
|---|---|---|---|
| MMVP / MoF | 视觉辨别 pair、encoder similarity | benchmark / 数据配方 | 官方代码可读；根 LICENSE 缺失，README 声明 MIT |
| VisOnlyQA | 几何感知评估、文本化对照 | benchmark / control | GPL-3.0；基础图像数据另有许可 |
| Hidden in Plain Sight | dense feature readout、深度 DPT、跨层比较、blank-image control | P1 readout | 官方代码可读；未找到仓库许可 |
| Visual Representations inside the LM | visual K/V readout、input-agnostic key 消融 | P1 论文重实现候选 | 未定位官方实现 |
| Cross-modal Information Flow | attention knockout、层窗口扫描、答案概率 | P0 intervention | MIT；LLaVA 实现 |
| VAR | LLM sink 检测、head 选择、knockout control、attention redistribution | P1 diagnostic；可选 intervention recipe | Apache-2.0；具体模型和版本约束 |
| DIYSink | ViT/LLM sink 分离、token-set 对照、isolated-token decoding | P1 diagnostic；训练架构 P2 | 官方项目页仍称代码即将发布 |
| SADT / HARM / VEED | attention-conditioned logit lens consistency、region masking、解码干预 | P1 组合方法 | **实时仓库已含实现**，Apache-2.0 |

## 1. MMVP / MoF

标题、arXiv ID 与 CVPR 2024 均确认。MMVP 的关键诊断是寻找 CLIP 表征相近、视觉区别明显的成对图片，再观察模型能否辨别；MoF 则是融合 CLIP 与 DINOv2 的模型方案。它适合提供干预样本和视觉混淆案例，不应把整个 MoF 模型注册为一个 probe。[正文](https://arxiv.org/html/2401.06209v2)、[CVPR 记录](https://openaccess.thecvf.com/content/CVPR2024/html/Tong_Eyes_Wide_Shut_Exploring_the_Visual_Shortcomings_of_Multimodal_LLMs_CVPR_2024_paper.html)

[官方仓库](https://github.com/tsb0601/MMVP)中 `LLaVA/llava/model/llava_arch.py::prepare_inputs_labels_for_multimodal_withdino` 是 MoF 接入点。README 声明 MIT，但所查根目录没有对应 LICENSE；嵌套 LLaVA 的许可不能代替整个仓库的声明。建议先做数据加载与相似度统计，记录 pair 的构造规则。paired accuracy 与单题 accuracy 要区分。

## 2. VisOnlyQA

标题、ID、COLM 2025 确认。它将几何感知任务与复杂语言推理尽量分离，提供训练/评估数据和文本输入对照。可用来衡量某个内部干预是否改变感知表现，但本身不是 activation 或 attention probing 算法。[正文](https://arxiv.org/html/2412.00947v3)

[官方仓库](https://github.com/psunlpgroup/VisOnlyQA)提供 `src/evaluation/evaluation.py`、`metrics.py`、`paired_bootstrap.py`。推荐接入 benchmark/control 层。README 明确提示 VLMEvalKit 转换版本与原版本不同，报告必须带上 dataset revision。[LICENSE.md](https://github.com/psunlpgroup/VisOnlyQA/blob/main/LICENSE.md)声明代码与数据 GPL-3.0，并列出原图来源的其他许可。

## 3. Hidden in Plain Sight

COLM 2025 由正式论文和作者仓库确认。可抽象的是：对视觉 encoder、projector 及后续层做直接 readout，再与同任务 VQA 输出比较；对应关系使用特征 cosine similarity，深度使用训练过的 DPT decoder；还有 blank-image 与 prompt 对照。因此必须记录 readout 是否训练、训练数据以及各层的空间映射。[正文](https://arxiv.org/html/2506.08008v1)、[COLM 论文](https://openreview.net/pdf/74c909720993b9aa6a3d8e2e6f98ea5bea6cec00.pdf)

[官方仓库](https://github.com/stephanie-fu/hidden-plain-sight)入口是 `eval_runner.py`，对应实现见 `evals/tasks/blink_correspondence.py` 与 `evals/models/probes.py::DPT`。环境含 PyTorch 2.4.1、Transformers 4.38.1 和 Prismatic。未找到明确仓库许可。

**初始 list 的映射需纠正：**“Visual Evaluation”是这篇论文图中的评估分支标签；未找到把它作为独立 COLM 2025 activation-patching 方法的证据。`prismatic_patch.py` 修改模型执行程序，并不等于 clean/corrupt activation replacement。建议暂把这条映射到 `VisualReadout`，不要伪造一个论文方法名。

## 4. Visual Representations inside the Language Model

标题、ID 与 COLM 2025 确认。论文将各层 visual K/V 当成可直接评估的表征，进行分割与对应关系任务；用跨图片方差识别 input-agnostic keys，再做定向 attention 阻断。还比较在图片之前添加文本 prefix 的效果。[正文](https://arxiv.org/html/2510.04819v1)、[COLM 论文](https://openreview.net/pdf/e56bd3ff7464a65655d5addd0f901a2e1c37cb4f.pdf)

截至核查，正文、[作者主页](https://liubl1217.github.io/)及 GitHub 精确题名检索未定位官方代码，许可和实现入口未知。可以纳入论文重实现候选，但应标记为 `paper_reimplementation`。适配上必须支持 GQA 的 query-head / KV-group 对应；input-agnostic 阈值要在目标模型校准。视觉 tokens 是否可读到文本，由 prefix 顺序和 causal mask 决定。

## 5. Cross-modal Information Flow

CVPR 2025 确认。主要方法是对特定层窗口中、特定 token 组之间的 attention edges 加 mask，测量目标答案概率变化，以定位视觉信息流。可复用原语为 `AttentionKnockout`，扫描维度是 layer window、source keys、destination queries，输出包含未干预与干预后的 target score。[正文](https://arxiv.org/html/2411.18620v2)

[官方 MIT 仓库](https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM)入口 `InformationFlow.py`；核心是 [`methods.py::set_block_attn_hooks_llava`](https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM/blob/main/methods.py)，另有 `last_position_answer_prob.py`。原代码直接替换 LLaVA attention forward，并区分 prefill 与单 token 解码。library 应用 context manager 恢复 hook，合并原 causal/padding mask，明确 query/key 方向。源码中虽有 `set_hs_patch_hooks`，论文主方法仍应标为 attention knockout。

## 6. VAR / Visual Attention Sink

ICLR 2025 确认。论文基于固定隐藏维度相对 RMS 的异常激活识别 LLM sink；比较 sink knockout 与等量随机 token knockout，并选择 image-centric heads 重分配 attention。建议拆成 `SinkDiagnostics`、`HeadSelector` 和可选 `AttentionRedistribution`。[正文](https://arxiv.org/html/2503.03321v1)、[ICLR 记录](https://proceedings.iclr.cc/paper_files/paper/2025/hash/da8a39bc39ae1c89dd6ebb1e3bcbb3f3-Abstract-Conference.html)

[官方 Apache-2.0 仓库](https://github.com/seilk/VisAttnSink)在 [`src/logic/logic.py`](https://github.com/seilk/VisAttnSink/blob/main/src/logic/logic.py)提供 `DimProspector`、`HeadFork`、`VARProcessor`；sink dimensions 位于 `src/logic/constants.py`，目前列出 Llama-v2 7B/13B。它不是架构无关即插即用实现。源码使用 class-level mutable state；新库应改为会话内状态。需保存 sink score 定义、阈值和随机消融方案，不能用 attention 大小直接当因果重要性。

## 7. DIYSink

ICLR 2026 接收记录由[官方 poster 页面](https://iclr.cc/virtual/2026/poster/10007059)确认，链接到 [OpenReview](https://openreview.net/forum?id=sQGlhjKUC0)；后者本次遇浏览器验证。arXiv 页面仍写 preprint，不能只据此否定会议归属。

论文区分 ViT 高 norm sinks、传播进 LLM 的 sinks 与 LLM 自身产生的 sinks；包含 sink-only/non-sink-only 对照和隔离视觉 token 后的词汇解码。ViT sink 可能含全局语义，不能统一解释成无效 token。sink-to-front 是推理干预；DIYSink 的双 projector 与 gating 是需训练的模型架构。[正文](https://arxiv.org/html/2510.08510v1)

[作者项目页](https://davidhalladay.github.io/diysink_demo/)仍标“Code (Coming Soon) Upon Acceptance”；本次 GitHub 检索只找到网站仓库，没有定位正式方法实现。建议先列 `ViTSinkDiagnostics` 与 token-set ablation recipe，源代码许可/入口暂空。sink-to-front 涉及位置映射，不能只重新排列 embedding 而不处理位置。

## 8. SADT / HARM / VEED

arXiv ID 与标题匹配，CVPR 2026 由[官方论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_Same_Attention_Different_Truths_Put_Logit-Lens_over_Visual_Attention_to_CVPR_2026_paper.pdf)确认。方法将高 attention 图像区域做 Logit Lens 解码，检查与生成对象词的一致性；再遮挡该区域，比较幻觉是否持续。HARM 是区域遮挡，VEED 是视觉证据引导解码。[正文](https://arxiv.org/html/2608.07302v1)

**代码状态以实时核查为准：**论文和搜索缓存写着 forthcoming，但 [SADT 当前仓库](https://github.com/wzczc/SADT)已含 Apache-2.0 实现，所查 revision 为 `e07a8432498334d91040ef9ede9371b32e6a8d3c`。核心 [`tools/sadt/core.py`](https://github.com/wzczc/SADT/blob/e07a8432498334d91040ef9ede9371b32e6a8d3c/tools/sadt/core.py)提供 `detect_hallucinations`、`mask_high_attention_regions_vcd`、`build_vcd_hallucination_guidance`；完整入口为 `tools/sadt/detect_classify_mitigate.py`。

当前代码假定 LLaVA 图像 marker `-200`、576 tokens、24×24 grid、336px，固定 Transformers 4.37.2；这些必须移到 adapter，不能扩散到方法类。还需记录 semantic matcher（COCO/WordNet/LLM 等）及层编号。其 readout 直接调用 `lm_head`，应与带 final norm 的通用 Logit Lens 分开配置。HARM 实际修改输入像素并重新生成，不能与 decoder attention knockout 混同。以上是源码审查结果，尚未声称复现论文指标。

## 对统一接口的直接要求

下列是本调研推导的设计约束：

1. `TokenLayout` 应描述图像 patch、prefix、question、answer 与空间坐标，支持可变 token 数和多图片。
2. `Site` 至少区分 vision feature、projector output、residual、Q/K/V、attention logits、attention probabilities；GQA 的 KV group 与 query head 单独建模。
3. `ReadoutProbe` 接收缓存表征与 task-specific evaluator；`Intervention` 接收 site、source/destination selector 与操作。用组合实现论文配方，避免每篇论文一个庞大模型类。
4. 结果保留 `observational` / `interventional`、层/头索引、target score、control、模型 revision、source revision。readout 可解码性不自动代表模型实际使用该信息。
5. 保留 `native_reference` 和 `generic_reimplementation` 两种身份；公开代码、可合法复用、已经适配、已经复现四个状态分开记录。
