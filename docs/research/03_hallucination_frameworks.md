# 幻觉传播分析与 probing 后端核验

核验日期：2026-09-17。范围为用户列表中的 OPERA、VISTA、MMHalSnowball/RVD、Visual Flow，以及四个通用解释工具。阅读了论文正文与官方仓库的关键源码；未安装这些工具、下载模型或复现论文指标。下述 class 名称属于本 library 的设计建议，不能据此理解为已经实现。

## 四篇论文的取舍

| 工作 | 身份/会议信息 | 可纳入的分析 | 在 library 中的定位 | 官方代码状态 |
|---|---|---|---|---|
| OPERA | CVPR 2024，arXiv:2311.17911 | 生成文本局部窗口中的 attention 聚集分数、summary-token 位置和随生成过程的轨迹 | attention diagnostic；完整 OPERA 为独立 decoding intervention | 代码公开，MIT；依赖改写的 Transformers 4.29.2 |
| VISTA | ICML 2025，PMLR 267，arXiv:2502.03628 | layer × generation-step 的目标词 rank；有图/无图 residual 差向量 | 优先纳入 lens trajectory；VSV/SLA 作为 steering/decoding 扩展 | 代码公开，MIT；架构与图像 token 假设需迁移 |
| MMHalSnowball / RVD | ACL 2024，2024.acl-long.648 | clean/factual/irrelevant/hallucinatory context 对照、FR/WFR；残余视觉与无图输出的 JSD | behavioral protocol + distribution diagnostic；RVD 是 mitigation | 数据与代码公开，GPL-3.0；采样函数使用全局 monkey patch |
| Visual Flow / ViF | arXiv:2509.21789；官方仓库与 OpenReview PDF 索引标为 ICLR 2026 | 跨轮/跨层/跨视觉 token attention；按趋势选 token，再做分层 token ablation | 纳入论文分析 recipe；完整 ViF 暂缓 | 官方仓库存在，但真实 VLM backend 为 stub，未发现仓库许可证 |

前三篇的会议与身份分别由 [OPERA arXiv 元数据](https://arxiv.org/abs/2311.17911)、[VISTA PMLR](https://proceedings.mlr.press/v267/li25ca.html)、[ACL Anthology](https://aclanthology.org/2024.acl-long.648/) 核对。ViF 的 [OpenReview 链接](https://openreview.net/forum?id=Ii4HBlERix) 与直接 PDF 在本次访问触发浏览器校验；搜索可见的 [OpenReview PDF 索引](https://openreview.net/pdf/a0c81d92bcb87e542f024c235b913e1c117ec355.pdf) 标明 ICLR 2026，且[作者官方仓库](https://github.com/xlyu0106/ViF)一致。正文核验使用 [arXiv v2](https://arxiv.org/html/2509.21789v2)，未将可访问的预印本当作已经比对过的会议终稿。

### OPERA：抽取诊断量，不把完整 beam-search 当作 probe

论文 §3.2 在已生成文本的局部 causal attention 窗口计算缩放后的列乘积，并以最大值表示聚集强度；先对多个 head 取最大值并重新归一化。图像与 prompt token 不属于这个局部窗口。建议抽成 `AttentionAggregationProbe(window, scale, head_reduce)`，返回每一步的聚集分数与候选 summary-token 索引。该量描述注意力聚集，不能直接当作某个 token 导致幻觉的因果证据。完整 OPERA 还包含候选前向、beam penalty 与回溯，属于生成算法。[论文正文 §3](https://arxiv.org/html/2311.17911v3)

官方主入口为 [`transformers-4.29.2/src/transformers/generation/utils.py`](https://github.com/shikiw/OPERA/blob/main/transformers-4.29.2/src/transformers/generation/utils.py) 的 `opera_decoding`；README 提供跨版本移植位置。建议独立实现公式级诊断，完整 decoder 放在 optional integrations，避免把旧版 Transformers fork 作为全库依赖。根目录为 [MIT](https://github.com/shikiw/OPERA/blob/main/LICENSE)，仓库还包含各自授权的上游代码，不能把整个 vendor tree 一律当作 MIT。[官方集成说明](https://github.com/shikiw/OPERA)

### VISTA：值得直接纳入的 token-rank trajectory

论文 §2.2 对 hidden genuine、decoded genuine、hallucinated 三类目标词记录各层、各生成步的 logit rank；其对象词标签由 GPT-4o 辅助构建。可通用化为 `TokenRankTrajectory(target_token_ids, layers)`，分类标签由调用者提供，不把外部标注服务做成运行依赖。VSV 是同一 prompt 有图与无图时最后位置的逐层 residual 差，生成中注入后保持原 hidden norm；SLA 混合早层与最终层 logits。两者是干预扩展，不能与只读 lens 混成一个类。[论文 §2、附录 A](https://arxiv.org/html/2502.03628v2)

关键入口：[`steering_vector.py`](https://github.com/LzVv123456/VISTA/blob/main/steering_vector.py) 的 `obtain_vsv`、`ForwardTracer`；[`llm_layers.py`](https://github.com/LzVv123456/VISTA/blob/main/llm_layers.py) 的 `VSVLayer`、`add_vsv_layers`；[`chair_eval.py`](https://github.com/LzVv123456/VISTA/blob/main/chair_eval.py) 演示添加与移除干预。官方代码把 image-specific steering 的 batch size 限为 1；[`anchor.py`](https://github.com/LzVv123456/VISTA/blob/main/anchor.py) 使用 LLaVA 576、MiniGPT4 32、Shikra 256 等固定视觉长度，不能直接套用动态分辨率 Qwen-VL。许可证为 [MIT](https://github.com/LzVv123456/VISTA/blob/main/LICENSE)。

### MMHalSnowball / RVD：行为对照和分布差异各自独立

MMHalSnowball 是 4,973 个样本的对照评测；FR/WFR 的分母应为 clean condition 下回答正确的样本数。可抽成 `ContextConflictProtocol` 和 `FlipMetrics`，按 sample ID 配对，保留 clean/factual/irrelevant/hallucinatory 四类上下文。RVD 则混合 `(image, history, query)` 与 `(image, query)` 的 logits，并用后者与 `(query)` 的 JSD 调节混合程度；这里的 residual 指输入路径，**不是 residual-stream activation patching**。[论文 §2–3](https://arxiv.org/html/2407.00569v1)

源码入口为 [`evaluation/eval.py`](https://github.com/whongzhong/MMHalSnowball/blob/main/evaluation/eval.py) 与 [`residual_visual_decoding/rvd_sample.py`](https://github.com/whongzhong/MMHalSnowball/blob/main/residual_visual_decoding/rvd_sample.py)。评测源码采用 `zip` 顺序配对，因此 library 应额外验证 ID 一致与覆盖率。README 使用 `evolve_rvd_sampling`，当前文件实际定义 `rvd_sampling`，不能照抄 README 即视作复现通过。采样函数替换 `GenerationMixin.sample`，需隔离适配。仓库授权为 [GPL-3.0](https://github.com/whongzhong/MMHalSnowball/blob/main/LICENSE)；当前仅记录引用，不复制其源码。

### Visual Flow：分析方法可用，官方仓库当前不等于完整复现

论文 §2 分析轮次、层与 token 的视觉 attention，按照 inactive/rise/fall/unimodal 趋势分组，在浅/中/深层去除不同份额 token。建议实现为 `AttentionTrajectory` 与 `TokenAblation` 的组合 recipe；输出每轮曲线与干预效果，并使用随机等量 token 作对照。完整 ViF 含视觉 relay token、轻量 transformer contextualization 和 attention reallocation，应归入模型/多智能体 intervention 扩展。[论文 §2–3](https://arxiv.org/html/2509.21789v2)

官方链接 `YU-deep/ViF` 现重定向至 [`xlyu0106/ViF`](https://github.com/xlyu0106/ViF)。实查 [`vif/models/vlm_iface.py`](https://github.com/xlyu0106/ViF/blob/main/vif/models/vlm_iface.py) 默认返回 stub 文本，真实模式直接抛 `NotImplementedError`；[`vif/models/base_stub.py`](https://github.com/xlyu0106/ViF/blob/main/vif/models/base_stub.py) 用 `torch.randn` 生成 hidden、attention 和 key states。`vif/utils/selection.py`、`reallocation.py` 可供理解接口，但不代表论文完整实现。GitHub repository metadata 未声明 license，递归文件树未发现 LICENSE；标记为 `license_missing` 与 `scaffold_only`，不得标为已验证可复用开源实现。

## 通用后端：截至核验日的实际能力

| 后端 | 适合的职责 | VLM 核验依据 | 许可证 | 采用建议 |
|---|---|---|---|---|
| TransformerLens | cache、命名 hook、activation / attribution / path patching | 当前 main 已有 TransformerBridge 与 LLaVA、Qwen3-VL、Qwen3.5 multimodal adapters；旧版纯文本结论已过时 | MIT | 可选后端，固定验证过的版本/commit，先做 HF 原模型前向一致性检查 |
| NNsight | 直接读取与替换 PyTorch 中间状态，trace/generate | 官方 0.6 发布文明确新增 VisionLanguageModel，源码有 VLM wrapper | MIT | 很适合通用 HF VLM 干预；trace 生命周期封装在 adapter 内 |
| pyvene | composable interchange、subspace、learned intervention | 官方 BLIP 教程及 LLaVA 映射文件 | Apache-2.0 | 可作为 DAS/可学习子空间干预的后端，不强制所有 probes 依赖它 |
| Captum | Integrated Gradients、LayerIG、Occlusion、FeatureAblation 等 | 支持 PyTorch 可微 forward；无证据表明自动处理所有 VLM token/grid 语义 | BSD-3-Clause | attribution 扩展后端；由 library 提供 VLM 输入、目标分数与 baseline 适配 |

TransformerLens：查阅 [README](https://github.com/TransformerLensOrg/TransformerLens)、[Qwen3-VL adapter](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/model_bridge/supported_architectures/qwen3_vl.py)、[LLaVA adapter](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/model_bridge/supported_architectures/llava.py)、[attribution patching](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/tools/analysis/attribution_patching.py) 与 [LICENSE](https://github.com/TransformerLensOrg/TransformerLens/blob/main/LICENSE)。这些是当前源码存在性证据，本次没有证明所有功能都已进入所安装发行版，也没有运行该库的 tests。默认 Bridge 与旧 HookedTransformer 的权重处理不同，比较实验应记录配置。

NNsight：依据 [0.6 官方发布说明](https://nnsight.net/blog/2026/02/26/introducing-nnsight-06/)、[`modeling/vlm.py`](https://github.com/ndif-team/nnsight/blob/main/src/nnsight/modeling/vlm.py)、[MIT LICENSE](https://github.com/ndif-team/nnsight/blob/main/LICENSE)。本地 trace 不需要启用 NDIF 远程执行；library 不应为了普通 probing 自动开启远程服务。

pyvene：依据 [官方项目](https://github.com/stanfordnlp/pyvene)、[Intervening on Vision-Language Models / BLIP](https://stanfordnlp.github.io/pyvene/tutorials/advanced_tutorials/Interventions_with_BLIP.html)、[LLaVA 映射](https://github.com/stanfordnlp/pyvene/blob/main/pyvene/models/llava/modelings_intervenable_llava.py)、[Apache-2.0 LICENSE](https://github.com/stanfordnlp/pyvene/blob/main/LICENSE)。支持 module hook 不等同于已经正确识别任意新 VLM 的图像 token 范围。

Captum：依据 [官方算法文档](https://captum.ai/docs/attribution_algorithms)、[代码仓库](https://github.com/meta-pytorch/captum)、[BSD-3-Clause LICENSE](https://github.com/meta-pytorch/captum/blob/master/LICENSE)。IG 必须指定可微 score 与有意义的 baseline；离散 `input_ids` 不能直接求导，应在 embedding/视觉张量或指定 layer 上解释。模型 eval mode 与梯度开启是两件事。attention weights、梯度 attribution 和 causal intervention 应在结果元数据中明确区分。

## 对 library 结构的直接建议

分析层只依赖统一 `Trace`、`TokenLayout`、`Score` 与 capability 声明。后端隐藏 PyTorch hook、NNsight trace、TransformerLens cache 的差异。token-rank 与 attention 轨迹是只读分析；VSV、token ablation 是干预；OPERA/RVD 是 decoding strategy；MMHalSnowball 是 evaluation protocol。保持这些职责独立，可复用相同 tracing 与结果序列化代码。

优先落地：`TokenRankTrajectory`、`AttentionAggregationProbe`、`AttentionTrajectory`、`TokenAblation` 和 `ContextConflictProtocol`。后续扩展：VSV 与三路输入的 JSD diagnostic。完整 OPERA、RVD、ViF 等应在各自模型/依赖合同通过后接入，不应为充实 method list 而提前宣称支持。
