# 可学习间隔发放与 Mamba 适配原型

日期：2026-09-10。独立模型：`integer-tcn-mamba2-interval`。
保留旧模型、数据和实验结果；本轮未启动正式训练或计算新 EER。

## 1. 与旧 Burst 前向策略的关系

沿用旧 `LearnableThresholdBurstPLIFTauChannelwiseNode` 的核心思路：
**正间隔 → 累加为有序阈值 → 累加超过阈值的次数**。

不复用旧实现的两个隐患：

- 不把基础电压阈值和无量纲阈值倍率混合求差；本版所有阈值都使用归一化电流的同一单位，
  从零开始累加，避免旧初始化中第一项偏移。
- 不令间隔无限增大，也不把每个阈值的恒等 STE 直接相加。

本节点没有跨时间 PLIF 状态或复位。记忆由 Mamba 的 SSM 承担，而不是再加一个逐步
脉冲反馈循环。未沿用旧单次前向里实际起幅度缩放作用的 tau。

## 2. 前向定义

每个正间隔采用有界可学习参数：

`gap_k = gap_min + (gap_max-gap_min) * sigmoid(raw_gap_k)`

`theta_k = sum(gap_1 ... gap_k)`

有符号发放：

`q(u) = sum_k [ H(u-theta_k) - H(-u-theta_k) ]`

其中 H 使用严格大于零。正负两侧共享阈值，故 `q(-u)=-q(u)`。
默认 K=D=4，输出 `{-4,-3,-2,-1,0,1,2,3,4}`，不除以 K。

- 初始间隔：`[0.5,1,1,1]`，初始正阈值约 `[0.5,1.5,2.5,3.5]`。
- 间隔范围：`0.05 ... 2.0`，所以阈值有序、有界。
- 初始和后续值允许正常浮点舍入误差。
- 半整数边界采用严格阈值，不宣称与 `torch.round` 的 ties-to-even 逐位相同。
- 默认每个插入位置共享一组 K 个间隔；可选逐通道间隔。
- `--interval-fixed` 冻结同一组间隔，作为匹配的固定间隔对照。

默认两层 Mamba、每层输入/输出两组间隔，总计只增加 **16 个可训练参数**。
逐通道模式会增加更多参数，不应与共享模式混作同一实验。

阈值是发放决策边界，输出仍然是等权事件计数；它不是返回不等间距的重构电压。
因此间隔学习改善的是计数量化边界的适应性，不自动保证连续值重构误差最小。

## 3. 反向估计

令阈值附近的三角窗口分别为 phi_pos_k 和 phi_neg_k，窗口宽度默认为 1：

`Z = max(1, sum_k(phi_pos_k + phi_neg_k))`

`dq/du ~= sum_k(phi_pos_k + phi_neg_k) / Z`

`dq/dtheta_k ~= (-phi_pos_k + phi_neg_k) / Z`

输入方向的总斜率被限制在 `[0,1]`，避免 K 或正负分支数导致梯度直接放大。
阈值梯度再经 cumulative sum 和 sigmoid 传到 raw_gap。
这是人为指定的替代梯度，不是硬阈值的真实导数，也没有“必然最优”的理论保证。
固定间隔对照仍使用同样的输入替代梯度，以免把梯度规则改变混入间隔学习对比。

实现分两遍扫描小的阈值数 K，避免构造 `[B,C,L,K]` 大张量。
没有对签名时间 L 的 Python 循环；不改变现有 SSD 的块间并行算法。
间隔参数在 AdamW/SGD 中都不施加 weight decay。

## 4. 插入 Mamba 的位置

```text
连续残差 h
  → LayerNorm
  → 有符号可学习间隔发放（input site）
  → Mamba 输入投影
  → 原因果短卷积、SiLU、B/C/dt、SSM、skip、门控 RMSNorm
  → 有符号可学习间隔发放（output site）
  → Mamba 输出投影
  → 连续残差相加
```

默认 `--interval-sites both`，也可分别指定 `input` 或 `output`。
保留旧的 8 个 TCN 发放节点；默认增加 4 个 Mamba 发放节点，共 12 个。
所有对外发放钩子保持 `[B,C,L]` 布局，支持有效长度与序列特征导出。

选择这两个位置的理由：输入已做通道归一化，输出已经过门控 RMSNorm，阈值单位较可控；
两处能让 Mamba 的两个线性投影直接接收有符号整数值，且不直接把衰减系数量化到很少级别。
**但输入量化仍会间接改变 dt/B/C，不能说控制信号与原模型完全不变。**

未直接量化：衰减系数、B/C 分支、内部记忆、短卷积中间值、门控函数、残差和读出。
这些部分及权重仍使用浮点。当前是整数值激活原型，**不是全整数算术或全脉冲 Mamba**。
推理时可固定学习好的阈值；若进一步导出为整数比较阈值，还需要重新验证定点精度与排序。

## 5. 适配性结论与已做检查

结构上兼容：新增量化只在投影边界，SSM 仍是输入驱动的仿射状态更新，当前并行计算可保留。
正负事件能进入投影，不像单侧发放那样直接将全部负输入归零。
这不等于已经证明 EER 会改善。

已经检查：

- 初始化、正负对称、硬整数输出、严格阈值边界、间隔顺序及上下界。
- 前向对应旧式阈值计数，反向对应声明的归一化替代梯度；重叠窗口不放大总输入梯度。
- Padding 不发放、不污染阈值梯度；支持固定/可学习、共享/逐通道间隔。
- Mamba 投影确实接收整数值；输入/输出位置可独立开启。
- 同种子下，原 Mamba 与间隔版本的已有参数初始化一致；固定/可学习版本初始输出一致。
- 因果性、不同长度、activation checkpoint、episode 梯度累积、模型/配置/检查点恢复。
- 旧模型和数据回归保持通过。

GPU 随机序列检查（RTX 4070 Laptop、FP32、batch=16、activation checkpoint，
CE 前向+反向，不更新权重；预热一次、测量一次，不能作为稳定性能统计）：

| 序列长度 | 峰值 allocated 显存 | 一次前向+反向 |
|---:|---:|---:|
| 1024 | 350.84 MB | 67.93 ms |
| 11819 | 3989.75 MB | 1499.22 ms |

输出与梯度均有限。该原型增加了量化与反向开销，并不承诺比连续 Mamba 更快。
最长随机输入下，Mamba 输入侧约 62% 位置非零，输出侧约 34%–35%；输出侧饱和约 1.5%。
这些是未训练模型的随机输入统计，不可当真实数据最终发放率或节能指标。

另从 DeepSignDB 的 MCYT **训练集**抽取一个真实 episode（16 条，最大长度 914），
使用 AP+CE 做了一次只更新间隔的临时 SGD 检查。四组 raw_gap 的梯度都有限且非零，
阈值均有变化（最大变化约 4.8e-7～4.8e-6）。没有保存这次临时权重，也没有进行正式训练。
正式 528 类配置总参数为 **392,552**，比连续 Mamba 版本多 16 个。

## 6. 配置、日志与后续验证

独立配置：`configs/integer_tcn_mamba2_interval.json`。
正式训练前仍建议先比较固定间隔与可学习间隔，不把一切变化归因于“学习”。
保持 DeepSignDB 集中式、同一真实数据目录、种子 1111、AP+CE、100 epochs、训练 loss 选权重，
不加 DTW 或新验证集。2026-09-11 已按用户要求启动可学习间隔正式实验，详见
`EXPERIMENT_RUNS.md`。以下为本次启动参数，正在运行时不要重复启动：

```powershell
python src/run_integer_tcn_experiment.py --config configs/integer_tcn_mamba2_interval.json --run-tag interval-mamba-deepsign-central-loss-s1111-20260911-v1 --seeds 1111 --device cuda --microbatch-tasks 1 --cpu-threads 4
```

固定间隔对照可复制配置为新文件，将 `architecture.interval_fixed` 设为 true，使用新 run-tag。
普通训练入口也支持 `--interval-fixed`。共享/逐通道及插入位置必须分别记录，不能偷偷混改。

训练运行器会保存 `initial_intervals.json`、每 epoch 更新的 `interval_diagnostics.json`，
以及最佳训练 loss 权重对应的 `best_train_intervals.json`；完整断点包含最佳间隔诊断快照。
阈值参数、间隔边界、插入位置和梯度版本都写入检查点元数据，恢复时严格校验。

源码：`learnable_interval_spike.py`、`network_interval_mamba.py`。
原 `network_snn.py` 的旧 Burst 神经元没有被修改；新模型与之前权重、数据、实验结果隔离。

最终回归：**112 passed**（27.49 秒），仅有 PyTorch TypedStorage 弃用提示。
包含实际训练入口/评估入口恢复、间隔梯度声明检查、初始/最佳间隔日志与断点修复。
