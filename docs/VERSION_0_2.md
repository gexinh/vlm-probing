# v0.2.0 版本总结

## 本版本目标

将已有 18 个张量算法封装成可以直接绑定模型、输入模型原生 kwargs 的 library。
用户只需初始化 `Prober(model, processor=...)`，从三个方法家族创建方法，再调用
`.run(inputs)`。需要校准的方法提供 `.fit/.save/.load`。仓库根目录改为英文 README，
说明用途、安装、示例、支持范围和方法含义。

## 与 v0.1.0 的差别

| 项目 | v0.1.0 | v0.2.0 |
| --- | --- | --- |
| 入口 | 调用者手动提取张量、编写 runner | Prober + LensMethods / AttentionMethods / CausalMethods |
| 算法 | 三个基类、18 个具体子类 | 原样复用算法层，增加全部 18 个方法的模型绑定工厂 |
| 模型能力 | 手动维护 hook 字典 | ModelSpec、describe()、明确的可观察/可编辑能力区分 |
| 自动适配 | 无 | 类注册、model.probing_adapter() 协议、版本限定的 HF Llama / Qwen2.5-VL |
| token 选择 | 手动索引 | all / visual / text / last_prompt / 显式位置与 mask |
| 输入对齐 | 调用者负责 | 检查展开序列、padding、token IDs、视觉网格；显式位置对齐选项 |
| Lens 校准 | 每层手动提取 teacher 和 hidden | 捕获与校准自动串联，逐层持有拟合状态，绑定 artifact 身份 |
| 干预 | 自行写模型重跑和评分 | 自动执行独立逐层干预、返回 baseline/intervention/effect |
| 评分 | 张量工具函数 | 保留原函数，并增加 TokenMargin 和 SequenceLogProb |
| 结果 | 单个算法输出 | 层轴、打包 token 坐标、配置和 metric 元数据、默认 CPU 保存 |
| 文档 | 中文框架说明 | 英文主页、API 和 adapter 指南、更新后的架构说明 |

保留原有 BaseMethod → 家族基类 → 算法子类继承结构。上层采用组合调用，避免为
每个模型和算法重复维护一套实现。当前版本没有增加表征分析类方法，没有新增模型
加载器、数据库、调度器或插件系统。

## 清理与兼容性

- 重写旧主页和当前架构文档，移除过时的“所有自动适配尚未实现”说明。
- 更新方法目录版本和公共工厂映射，保留每个方法的算法边界。
- 原始张量 API 和旧例子继续可运行，避免同一算法在两处维护。
- v0.1.0 总结及此前文献调研作为明确的历史资料保留。
- 包构建和隔离安装验收使用临时目录，不将构建产物加入源码目录。

## 预期实验结果

以下是本版本的工程验收目标，不是对真实 VLM 准确率的预测：

1. 最终层 Logit Lens 与模型最终 logits 在相同 token 位置一致。
2. self-patch、零强度 steering、单位 attention 权重和单位温度保持输出不变。
3. 公共接口与直接 hook 干预一致；不同层的扫描保持独立实验语义。
4. 小幅干预的一阶估计接近 exact patch；积分归因满足数值容差内的完整性检查。
5. Tuned/Attention Lens 校准 loss 下降，不改变原模型权重、已有梯度或训练模式。
6. Jacobian 在最终 residual site 上恢复单位映射；fitted artifact 能往返加载且拒绝错配。
7. 不同 visual-token 数量和左右 padding 有正确坐标；非法 mask、错配 token/grid 和
   不支持的 attention 编辑被明确拒绝。
8. HF Qwen2.5-VL 走通实际视觉编码器和投影器；六种 Lens 与 residual patching 能在
   小型随机配置上执行。HF 支持范围不依赖“大模型论文指标已复现”的假设。

## 实现边界

- 自动 HF 适配限定 Transformers 4.57.x，本次验证版本 4.57.6；其他架构/版本可提供显式 adapter。
- 原生 HF 的 attention 重加权、温度、knockout 和 EAP-IG 不声明为支持。对应方法
  在完整 TinyModel adapter 上运行，实际模型需要暴露真实可编辑的计算位置。
- EAP-IG 是显式计算边上的 activation-space IG 变体；Attention Relevance 是指定
  self-attention 梯度传播；Patchscope 为 teacher-forced readout；EmbedLens 不包含
  整篇论文的聚类/剪枝实验。
- 没有运行预训练模型 benchmark、大规模 Lens 校准、视频、量化、多卡分片或生成缓存实验。
- calibration 接收单个批次；Tuned/Attention 重复 fit 保留权重但重置优化器，Jacobian
  fit 替换上次估计。artifact 身份由调用者声明，不伪造自动 checkpoint hash 校验。

## 验证记录

- **71 个测试全部通过**：原有算法/adapter 测试 45 项，新增公共 API 测试 22 项，
  可选 HF 架构集成测试 4 项。完整套件在隔离环境中的运行时间约 4.34 秒。
- 测试环境：Python 3.12.8、PyTorch 2.6.0、Transformers 4.57.6，全部使用 CPU。
- `examples/prober_quickstart.py` 跑通 18 个公共方法；Tuned Lens 演示 KL 从
  0.032694 降到 0.001925。这仅验证校准流程，不是 held-out 或论文结果。
- `examples/hf_smoke.py` 跑通 Llama 和 Qwen2.5-VL 六种 Lens 与观察/干预路径。
  Qwen 使用实际 vision encoder、projector 和 decoder，输入合成像素与不同图片网格；
  没有下载预训练权重。
- README 的 CPU 快速开始、校准、保存/加载，以及 ADAPTERS 的完整示例已执行通过。
  当前文档相对链接、Python 语法及 18 个机器可读工厂映射已检查。
- 旧 `quickstart.py` 和 `all_lenses.py` 均通过，保持 v0.1 的张量接口可用。
- 最终源码的 wheel 已构建，并在临时独立安装目录中跑通全部 18 个公开方法；
  该环境未安装 Transformers，验证核心依赖仅 PyTorch。
- 源码目录没有残留 build、dist、egg-info 或 __pycache__。

复查增加了两项实际兼容性保障：独立 readout/校准也在 eval 模式执行并恢复原模式；
forward 默认参数由 ModelSpec 显式声明，避免向普通自定义模型误传 HF 参数。
校验结果覆盖功能与接口，不声称已发布远程仓库或复现预训练模型论文指标。

## 验证命令

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python examples/prober_quickstart.py
PYTHONPATH=src python examples/quickstart.py
PYTHONPATH=src python examples/all_lenses.py
# 安装 optional Transformers extra 后：
PYTHONPATH=src python examples/hf_smoke.py
```
