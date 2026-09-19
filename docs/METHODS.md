# 当前方法与实现范围

v0.2.0 保留 18 个具体算法子类：Causal / Attention / Lens 各 6 个，并为全部方法增加模型绑定入口。本文描述底层张量 API；日常使用请看英文 [README](../README.md) 和 [公共 API](API.md)。以下是代码语义，不代表已经复现预训练模型上的论文数值。历史论文及官方仓库链接见 [catalog/methods.json](../catalog/methods.json) 和 [research](research/04_lenses_and_additions.md)。

## Lens：全部六种纳入

| 子类 / 文件 | 主要 API | 已实现 | 本版范围 |
|---|---|---|---|
| `LogitLens` / `lenses/logit.py` | `LogitLens(readout).run(hidden)` | 使用真实 norm/head 的逐 site 词表解码 | 层循环与 token 选择由调用者执行 |
| `EmbedLens` / `lenses/embed.py` | `EmbedLens(input_embeddings).run(hidden, top_k=...)` | vocabulary 分块 cosine 近邻、norm、可选显式 token 分组 | 官方 semantic readout；不包含整篇聚类/剪枝实验 |
| `TunedLens` / `lenses/tuned.py` | `.fit(hidden, teacher_logits)` / `.run(hidden)` / `.save/load` | identity-init residual affine、KL 拟合、模型绑定 | 每层/site 一个实例；大型语料训练由实验脚本组织 |
| `AttentionLens` / `lenses/attention.py` | `.fit(heads, teacher_logits)` / `.run(heads)` / `.save/load` | 可学习 per-head vocab maps、summed-logit KL、last-valid-token mask | 输入默认 post-W_O；显式传 unembedding 可使用论文式初始化 |
| `JacobianLens` / `lenses/jacobian.py` | `.fit(hidden, downstream, masks...)` / `.run(hidden)` / `.save/load` | 平均 causal source→final Jacobian、有效位置约束、source sampling | exact 输出维度循环，非大型模型高效拟合器 |
| `Patchscope` / `lenses/patchscope.py` | `.run(hidden, source_position, target_position, target_layer, target_inputs)` | 源 token 选择、可选 mapping、通过真实 runner 写入目标 prompt 并读出 | teacher-forced logits；自由生成及 KV-cache 后端未接入 |

Tuned/Attention 的 `fit` 返回每步 loss，Jacobian 的 `fit` 返回自身。训练只对 Lens 参数求梯度，不修改原模型 gradients/权重/模式。Tuned 可用 `readout_dtype` 在 FP32 translator 与 BF16 readout 之间显式转换。实际输入仍需位于合适的 device。

Jacobian 默认排除前 16 个有效 token 和最后一个有效 token，与本次核验的官方 reference convention 对齐；短序列示例显式设置 `skip_first=0, exclude_last=False`。估计器先累加有效当前及后续目标位置的作用，再平均每个 prompt 的有效 source，最后等权平均 prompts。下游回调必须保持 causal 性、batch 独立性和 token 对齐。它不是对单个候选 logit 求一次 saliency。

EmbedLens 不把 input embedding 换成 lm_head。其 token_groups 是调用者给出的 checkpoint-specific 规则；未匹配值为 -1，不自动宣称所有其他 token 都有语义。AttentionLens 与 HeadLogitAttribution 分别属于拟合读出和指定线性分解，不相互替代。

## Causal：六种子类

| 子类 | API 核心输入 | 返回 / 语义 |
|---|---|---|
| `ActivationPatching` | `run(receiver, source, mask=...)` | 选择位置的 replacement；不原地改动输入 |
| `Ablation` | `run(activation, mode='zero/mean/resample', reference=...)` | mean/resample 要求明确 reference；resample reference 是已对齐的参考张量 |
| `AttentionKnockout` | `run(logits, blocked=...)` | 合并已有 -inf mask，返回 edited logits / probabilities；整行被阻断时报错 |
| `AttributionPatching` | `run(receiver, source, gradient=... 或 metric=...)` | receiver 点的 delta×gradient，正方向为 source−receiver |
| `EAPIG` | `run(receiver, source, edge_names=..., metric=..., steps=...)` | caller-defined computational edge activations 上的直线 IG、per-edge score、completeness error |
| `Steering` | `run(activation, direction, strength=..., preserve_norm=...)` | 加方向，按显式规则选择是否保持 norm |

编辑类可传 `runner` 计算真实 baseline/intervention/effect。未传 runner 只表示已构造编辑张量，不标为已测量的因果效应。mask 按 PyTorch 广播规则验证，典型 token mask 是 `[B,S,1]`；不暗自补齐 shape 或跨模型映射。

`EAPIG` 明确是 **simultaneous edge-activation-space IG** 变体。调用者必须提供可单独替换的实际计算边及可微 graph replay。本版不自动抽取 transformer circuit，也没有实现 upstream 的 input-embedding interpolation 路径。不能给相关图或普通 node cache 随便添加 edge_names 后就宣称已完成原论文 EAP-IG。

## Attention：六种子类

| 子类 | 输入与结果 | 计算约定 |
|---|---|---|
| `AttentionProfile` | `[B,H,Q,K]` → entropy、concentration、modality mass | group masks 按 batch 给出，合法行概率和为 1 |
| `HeadLogitAttribution` | post-W_O heads `[B,H,Q,D]`、readout weight → `[B,H,Q,V]` | 使用全 residual 的固定 norm scale，norm gain 已折入 readout；bias 单列 |
| `AttentionRollout` | `[L,B,H,N,N]` → 累积 token 传播 | 显式 residual 与 head reduction；仅 square self-attention |
| `AttentionRelevance` | attention stack + 对目标 score 的 gradients → relevance | 正梯度加权 self-attention residual propagation 变体；不声称完整 Chefer 多模态/encoder-decoder 实现 |
| `AttentionReweight` | probabilities + weights → edited probabilities | 是否重新归一由构造参数明确；需写回真实 value aggregation |
| `AttentionTemperature` | masked logits → scaled logits / probabilities | 正的有限 T；保留 causal/padding -inf mask；不是 generation sampling temperature |

query-head 轴与 KV-head 轴不同。rollout/relevance 不是因果干预；attention 大小也不等于 causal importance。编辑 probability 后必须重新进行 A@V 及后续模型计算。所有范例和测试使用实际计算路径验证这一点。

## 当前不复现的内容

LinearProbe、RepresentationSimilarity、InputAttribution、SparseConceptProbe、GraphProbe 已从活动方法列表与代码框架移除，仅保留历史调研。需要这些方法的计数/表征实验也不属于当前交付。benchmark 数据读取、付费 judge、完整 OPERA/RVD/VEED decoder、DIYSink/ViF 模型改造不进入三个基类。

## 复现分层

- 代码框架：三个抽象分类基类、18 个具体算法子类、共用 adapter/result/metrics，已实现。
- 数值验证：CPU 单元语义测试、两个真实小模型 forward 示例；结果见 [VERSION_PLAN](VERSION_PLAN.md)。
- 模型集成：支持显式 ModelSpec、自定义 adapter 注册，以及 Transformers 4.57.x 的 Llama / Qwen2.5-VL 自动绑定。已运行随机小模型，包括 Qwen2.5-VL 实际图像编码和不同视觉网格；原生 HF attention 编辑和自动电路提取不属于已支持能力。LLaVA 等其他架构需显式适配。
- 论文实验：没有下载模型/数据集，没有运行大型 Lens 校准或论文 benchmark，因此没有“论文指标复现成功”的结论。
