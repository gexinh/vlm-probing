# v0.1.0 版本总结

## 本版本目标

根据用户修正，把 library 范围收敛到 Causal、Attention、Lens 三类；每类一个抽象基类，具体算法逐一继承并单独成文件。六种 Lens 全部进入实际代码，实现读出、拟合、Jacobian 校准和 Patchscope 目标 prompt 执行接口。

## 与 v0.0 的区别

| 项目 | v0.0 | v0.1.0 |
|---|---|---|
| 交付 | 文献目录和架构草案 | 可安装 Python 包、实现、示例、测试 |
| 继承组织 | 通用 Method Protocol 的设计 | BaseMethod → BaseCausal / BaseAttention / BaseLens → 18 个具体子类 |
| 活动范围 | 23 个候选方法含表征分析 | 18 个方法，三个家族各 6 个 |
| Lens | Logit/Embed 优先，其余后续 | 六种全部有代码：Logit、Embed、Tuned、Attention、Jacobian、Patchscope |
| 模型执行 | 拟议 Session/adapter/token map | 实际 TorchModelAdapter，显式 module site、scoped hooks、模式恢复 |
| 当前不做 | 尚未严格分离 | LinearProbe、Similarity、InputAttribution、SAE、Graph 已移出活动目录 |

删除了旧架构中尚未实现的 Session、自动模型适配、表征类和研究模块的活动计划，避免文档与实际代码不一致。保留最初 18 篇论文及额外工具调研作为历史依据。没有复制旧 Transformers fork，没有新增无算法的占位子类。

## 预期实验结果

这些是方法验收目标，不是对真实 VLM 准确率的预言：

- self-patch 与 no-op hook 应保持输出不变；改变已知相关激活或真实 attention edge 应影响 downstream score。
- final-layer LogitLens 与模型 readout 应一致；EmbedLens 的分块近邻应与完整 cosine 搜索一致。
- Tuned/Attention Lens 拟合应能在受控校准例子上降低 KL，同时不改变原模型权重和已有 gradients。
- JacobianLens 在解析可求的因果 continuation 上应得到正确平均矩阵；位置 mask 和 source sampling 应排除无效位置。
- Patchscope 的源 token 应实际写入目标计算，返回目标 prompt 的读出结果。
- artifact 换模型绑定应拒绝加载；错误位置/全部 masked attention/原地污染 gradient capture 应明确报错。

## 已验证结果

- **45 个 CPU 语义测试通过**：adapter/metric 10、causal 10、attention 9、lens 16。
- `examples/quickstart.py` 跑通：最终层 LogitLens 与真实输出一致；视觉 token patch 改变 margin；knockout 对真实 A@V 输入的边生效。
- `examples/all_lenses.py` 跑通全部六种 Lens；固定小模型示例的 TunedLens 校准 KL 从约 0.025195 降至 0.002570。该数字只验证实现路径，不是 held-out 或论文 benchmark 结果。
- 离线 wheel 构建、隔离目录安装和导入通过；验证三个分类基类不可实例化、18 个方法都是各分类基类的直接具体子类。
- 环境：Python 3.12.8、PyTorch 2.6.0；测试均在 CPU，不需要 pytest/Transformers，也未下载模型。

复查还修复了两处容易污染结论的问题：gradient capture 后的下游原地修改，以及不计分的 all-negative-infinity logits 使 answer score 梯度出现 NaN。

## 当前边界

这是方法计算与 library 框架版本，不是论文 benchmark 复现报告。没有运行预训练 VLM、大规模 lens calibration、GPU 实验或论文数据集。EAPIG 是 caller-defined edge-activation IG 变体，AttentionRelevance 是指定 self-attention gradient propagation 变体；Patchscope 当前返回 teacher-forced target logits。真实模型的视觉 token/grid 对齐、head sites、fused attention 编辑和生成缓存需要后续逐模型验收。

## 验证命令

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python examples/quickstart.py
PYTHONPATH=src python examples/all_lenses.py
python -m pip wheel . --no-index --no-deps --no-build-isolation
```
