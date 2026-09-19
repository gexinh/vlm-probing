# 初始方法校正、补充方法与已有工具库

核验日期：2026-09-17。以下区分论文提出的方法、公开代码、拟议的本库接口。代码存在不等于已经在本机运行通过。仓库许可与部分 commit 记录见 `catalog/repository_snapshot.json`；GitHub API 后半程发生限流，缺失的 commit / tree 保持未核验，部分许可另由 raw LICENSE 核实。

## 初始清单的校正

| 用户条目 | 核验及收录决定 |
|---|---|
| Visual Evaluation, COLM 2025 | 尚未确认这是独立论文名。可定位到 Hidden in Plain Sight 图中的评测标签；该文应收录视觉读出与表征利用诊断，不能据此声称它提出 activation patching。详见逐篇报告。 |
| Prismatic VLMs, ICML 2024 | 模型、训练和评估框架；作为 adapter / 受控模型来源，不单列 probing algorithm，也没有依据声称不同 checkpoint 的 activation 可以直接互换。 |
| Attention Knockout, CVPR 2025 | 对应 Cross-modal Information Flow；正式收录路径干预，但与 activation replacement 分开。 |
| Logit Lens, 2020 | 收录无训练读出。经过模型实际 final norm 和 unembedding，不把 embedding 相似度当成同一种方法。 |
| Attention Lens, BlackboxNLP 2023 | 收录需要拟合的 head-specific decoder。2023 workshop program 可核验，论文为 arXiv:2310.16270；不是普通 attention heatmap，也不是任意 head output 直接乘 lm_head。 |
| Jacobian Lens, Transformer Circuits 2026 | 官方代码已公开，Apache-2.0，但 README 明确标为不维护的 reference implementation。收录研究扩展，VLM 需要重新定义校准数据和位置集合。 |
| EmbedLens, CVPR 2026 | 官方代码已公开，Apache-2.0。读取 projected visual token 与 input embedding 的近邻，并结合聚类和干预分析 sink / dead / alive。收录近邻读出和 token 诊断，类别规则按 checkpoint 配置。 |

Prismatic 的原始来源：[论文](https://arxiv.org/abs/2402.07865)、[官方仓库](https://github.com/TRI-ML/prismatic-vlms)。许可 MIT；实际模型入口 `prismatic/models/vlms/prismatic.py`。它的设计空间实验支持受控比较，但“受控比较”和“跨模型 activation patching”是不同实验。

## Lens 的实现边界

### Logit Lens

定义为 `lm_head(final_norm(h_layer))`，实际顺序由 adapter 依据模型 readout 确定。最终层已经归一化的 hidden state 不能再 norm 一次。对 vocabulary top-k 的读取是表征读出，不能仅凭结果推断因果重要性。

来源：[2020 原始文章](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)、[Tuned Lens 中的实现](https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/nn/lenses.py)。

### Tuned Lens：建议补充，P1

按层拟合 affine translator，使中间表示的读出接近最终输出分布。适合比较原始 Logit Lens 与校准后的读出。训练模型权重保持冻结，translator artifact 必须绑定 checkpoint、层、norm、tokenizer 和校准分布。现成纯文本 translator 不自动兼容 VLM。

来源：[论文](https://arxiv.org/abs/2303.08112)、[MIT 官方实现](https://github.com/AlignmentResearch/tuned-lens)。入口 `tuned_lens/nn/lenses.py`，包含 `LogitLens` 和 `TunedLens`；建议可选依赖或小 adapter，不复制完整训练工具链。

### Attention Lens

为各 attention head 拟合词表读出。核验了官方 `attention_lens/lens/registry/lensA.py` 的 per-head linear maps，以及 `attention_lens/train/lightning_lens.py` 的分布拟合目标。该仓库原始训练流程主要围绕 GPT-style head output，移植需要明确 pre/post output projection 的张量语义，不能依照名字猜轴。

来源：[论文](https://arxiv.org/abs/2310.16270)、[MIT 官方实现](https://github.com/msakarvadia/AttentionLens)、[workshop program](https://aclanthology.org/2023.blackboxnlp-1.0.pdf)。拟议 `AttentionLens.fit/apply` 与无训练的 `HeadLogitAttribution` 分别注册。

### Jacobian Lens

官方 estimator 平均 residual-to-final-residual Jacobian，再由模型 readout 解码。源码 `jlens/fitting.py` 会把当前及后续有效 target positions 的作用求和，再对 source positions / prompts 平均；这与“给一个 token 求一次 gradient”不同。拟合需要多次 backward，不能用普通 detached activation cache。

来源：[原文](https://transformer-circuits.pub/2026/workspace/index.html)、[Apache-2.0 官方实现](https://github.com/anthropics/jacobian-lens)。拟议集成先支持加载匹配的 artifact，再支持分片拟合。校准时要记录 image/text token 集合和 estimator，纯文本估计器的 VLM 外推列为待验证。

### EmbedLens

在 input embedding 空间计算近邻，包含向量尺度处理及聚类分析。不能把 `lm_head` 权重替换进去并仍称论文 EmbedLens。仓库入口 `semantic_identification.ipynb`；干预位于修改过的 LLaVA 和 `eval_scripts/`。公开示例中的 `1141/26673/30296` 是特定 tokenizer 的 ID，不能跨模型写死。

来源：[论文 §3](https://arxiv.org/html/2603.00510v1)、[Apache-2.0 官方仓库](https://github.com/EIT-NLP/EmbedLens)。matched-size token removal 与跳过某些 sublayer 更新不是同一干预，实验配置必须保存具体语义。

## 新增的可复用方法

以下优先级是本项目的工程判断，不是论文结论。P0 为首版，P1 为首版之后，P2 为研究扩展。

| 方法 / 工作 | 可提取的能力 | 官方代码与许可核验 | 决定及主要限制 |
|---|---|---|---|
| Attention Rollout / Flow, ACL 2020 | 组合逐层 attention，形成输入 token 到输出位置的描述性路径 | [attention_flow](https://github.com/samiraabnar/attention_flow)，许可本轮未完整核验；[论文](https://aclanthology.org/2020.acl-main.385/) | P1；保留 residual / head aggregation 定义；不能作为因果证明 |
| Generic Attention-model Explainability, ICCV 2021 | gradient-weighted attention / 多模态 relevance propagation | [Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability)，MIT；[论文](https://arxiv.org/abs/2103.15679) | P1；需可微 attention；现代 VLM 需 adapter，不能照搬旧 LXMERT |
| Patchscopes, ICML 2024 | 把源表示写入解释 prompt，通过生成读出多 token 概念 | [官方 code 子目录](https://github.com/PAIR-code/interpretability/tree/master/patchscopes/code)，仓库 Apache-2.0；[论文](https://arxiv.org/abs/2401.06102) | P1；先同模型 patchscope，跨模型需显式已验证的 alignment / mapping |
| Attribution Patching / EAP-IG | 用 gradient×activation difference 筛选节点或边，再验证 circuit faithfulness | [EAP-IG](https://github.com/hannamw/EAP-IG)，MIT；`src/eap/attribute.py`, `graph.py`, `evaluate.py` | P1；近似分数需 exact patching 复核，图必须覆盖相关视觉输入路径 |
| CoX-LMM, NeurIPS 2024 | dictionary learning 多模态概念，词和图像的共同解释 | [XL-VLMs](https://github.com/mshukor/xl-vlms)，MIT；README 链接原论文 | P2；概念字典依赖模型、层及训练数据 |
| Representation Shift Steering, ICCV 2025 | 比较微调前后表示差异；以 shift vector 干预输出 | [XL-VLMs](https://github.com/mshukor/xl-vlms)，MIT | P1 中共享 `Steering` 干预；训练前后比较保存模型对齐条件 |
| Sparse Autoencoders Learn Monosemantic Features in VLMs, NeurIPS 2025 | 分解 vision encoder 特征，并做 concept-level intervention | [sae-for-vlm](https://github.com/ExplainableML/sae-for-vlm)，公开源码，许可证尚未确认 | P2；不是所有 decoder 层都能复用其 SAE |
| VL-SAE, NeurIPS 2025 | visual / linguistic alignment 的共享稀疏概念空间 | [VL-SAE](https://github.com/ssfgunner/VL-SAE)，公开源码，许可证尚未确认；[论文](https://arxiv.org/abs/2510.21323) | P2；区分 CLIP alignment probe 与生成式 VLM 的 decoder probe |
| Line of Sight | 线性概念读出、SAE 和 steering | [multimodal-saes](https://github.com/multimodal-interpretability/multimodal-saes)，许可未确认；入口 `sae/sae-trainer.py`, `steering/steering_rollouts.py` | P2 参考；现有大规模训练不是核心库安装要求 |
| Structural Graph Probing, CVPR 2026 | neuron co-activation graph、hub-neuron intervention、modality correlation | [vlm-graph-probing](https://github.com/he-h/vlm-graph-probing)，MIT；[论文](https://arxiv.org/abs/2603.27070) | P2；图构造 / GCN 诊断 / 真正模型干预分别记录，统计相关边不等同计算图因果边 |

其中 Graph Probing 的代码入口包括 `model.py`, `probing/extract_graphs.py`, `probing/hub_neurons.py`, `probing/intervene_neuron.py`。本轮读到模型提取及图构造代码；没有运行其训练或验证其跨模型准确率。

## 直接相关的已有 library

| 工具 | 已核验的定位 | 本项目如何使用 |
|---|---|---|
| [VLM-Lens](https://github.com/compling-wat/vlm-lens), EMNLP 2025 Demo, Apache-2.0 | YAML 配置、model-specific classes、hidden-state 提取及持久化；核验 `src/models/base.py` | 优先参考 model adapter 和输入处理；本库侧重 readout、causal intervention 和共同实验协议，不重复建一套数据库服务 |
| [MultiModal-Lens](https://github.com/AKHegde22/MultiModal-Lens) | README 提供 `HookedVLM/run_with_cache/run_with_hooks` 与多架构路由；README 称 MIT，但实时根 LICENSE 内容是 Apache-2.0，需上游澄清 | API 参考，不能把 README 的“20+ families”当本项目已验收兼容性；只列候选，不承诺成熟依赖 |
| [ViT-Prisma](https://github.com/Prisma-Multimodal/ViT-Prisma), MIT | vision/video transformer hooks、SAE 训练及公开 SAE | 用作 vision-tower backend / SAE artifact 来源，不能直接代替完整 VLM 的多模态 token map |
| [XL-VLMs](https://github.com/mshukor/xl-vlms), MIT | multimodal concept discovery、representation shift 与 steering | 参考可复用分析算子和论文 recipes |
| [LVLM-Interpret](https://github.com/IntelLabs/lvlm-interpret) | input-output relevance 和图文 attention 可视化，详见 Insight Over Sight 核验 | relevance baseline / 展示参考；先确认模型兼容性和许可证，再考虑移植 |

通用后端 TransformerLens、NNsight、pyvene、Captum 的当前状态单列于 [03 报告](03_hallucination_frameworks.md)。特别是 TransformerLens 当前已有 multimodal adapter 工作，不能根据旧论文的局限描述认定它仍只支持纯文本。

## 检索范围与结论强度

检索采用用户原始标题和 ID、官方 venue 页面、作者项目页、GitHub README 与具体代码文件交叉核验。额外候选重点覆盖 fitted lenses、gradient/circuit、concept/SAE、vision-tower 与现成 VLM 工具库。这是有边界的工程选型调研，不声称穷尽截至本日全部论文。

方法首版应覆盖“可读出什么、关注哪里、改动后是否影响答案”三个互补问题。新论文优先复用基础算子组合成 recipe，只有新增计算语义才增加核心 class。尚未运行的上游代码、不明确的许可、仅论文可查的方法都保留各自状态，不能合并写成“已支持”。
