# Integer-TCN：方案记录与实现说明

记录日期：2026-09-10。状态：已实现独立模型；尚无正式数据集训练结果。

最新数据方案：第一阶段只做 DeepSignDB stylus 的集中式真实熟练伪造训练。
默认目录为 `data_real_forgery/deepsigndb_stylus_v1`，训练范围与 DsDTW 公开代码对应。
详见 [DeepSignDB 集中实验与 ANN baseline](DEEPSIGNDB_CENTRAL_BASELINES.md)。
MSDS 副本和独立配置保留，后续再用；旧目录及旧模型默认路径保持不变。

## 1. 本次确定的方案

采用 **SpikeYOLO-inspired 单步整数发放 + 小型多尺度残差 TCN**。

- 模拟步数 S=1；签名采样点数量记为 L，二者不混用。
- 不沿 L 进行膜电位、复位或自适应状态递推，不重复整段输入。
- 时间上下文由因果扩张卷积提取，整段时间维交给卷积算子并行处理。
- 默认 D=4，输出计数为 {0,1,2,3,4}，不除以 D，不展开二值事件槽。
- 输入投影、归一化、残差和嵌入读出是连续值；块内特征卷积接收整数值。
- 它是混合/量化约束的 TCN，不是完整递归 I-LIF 的等价并行实现。
- 现有 `ILIFComparisonNode` 是 inhibitory LIF，与这里的 integer-valued I-LIF 不同。

取舍：相对于先前讨论的真实时间递推 APLIF，放弃神经元反馈记忆以优先保证
长序列训练效率；相对于二值 PSN，增加整数表达级别以测试量化误差是否减小。
不同时叠加 PSN、可学习 tau、发放适应、多阈值 Burst、图网络或 Mamba。

## 2. 发放函数及梯度

`q(u) = round(clamp(u, 0, D))`。

反向为 clipped STE：输入位于 `[0,D]` 时导数为 1，范围外为 0。
边界包含 0 和 D；round 使用 PyTorch 的 ties-to-even 行为，与参考源码一致。
改变 D 不会使整体量化器的反向斜率自动变成 D。
梯度设置局部封装，不依赖旧 `SurrogateSpike.gradient_mode` 的全局状态。

Norm 的通道级 affine 参数提供量化前的缩放与偏移。当前不包含自动阈值校准、
发放率正则或整数权重训练。观测到高饱和/静默比例后，再进行受控消融。

参考来源（本次仅借鉴算法，不引入上游依赖）：

- [SpikeYOLO 作者仓库](https://github.com/BICLab/SpikeYOLO)
- [整数训练与 mem_update 源码](https://github.com/BICLab/SpikeYOLO/blob/main/ultralytics/nn/modules/yolo_spikformer.py)
- [二值推理版本](https://github.com/BICLab/SpikeYOLO/blob/main/ultralytics/nn/modules/yolo_spikformer_bin.py)

## 3. 架构

输入沿用当前 12 维动态特征，优先使用 `[B,L,12]`。也支持 `[B,12,L]`，
当两维都为 12 时拒绝歧义，调用方应通过额外右侧 Padding 消除歧义并提供真实长度。

1. 因果 Conv1d：12 → 128，k=7，stride=1。
2. 四个残差块：宽度 128，扩张率 1、2、4、8，每块两个整数发放节点。
3. Masked mean + population std：128 → 256。
4. Linear：256 → 256；L2 normalization 得到验证嵌入。
5. 无偏置身份分类头，只用于训练。

单个块：

```text
h → LayerNorm(channel) → Q_D → Depthwise Causal Conv(k=5,d)
  → LayerNorm(channel) → Q_D → Pointwise Conv(k=1)
  → learned residual scale → add h
```

每个块只有一个 k=5 时间卷积，另一个是 k=1 通道卷积。因此默认最大卷积感受野：

`R = 7 + (5-1) × (1+2+4+8) = 67` 个采样点。

100 Hz 下首尾间隔为 0.66 秒。这与早期“两次时间卷积/块”的 APLIF 草案不同，
不能沿用其 127 点感受野。最终统计池化覆盖整条有效签名，但不是顺序敏感的长程记忆。
如果需要更长上下文，单独调整 kernel/dilations 并记录验证结果。

## 4. Mask、数值与状态

- 无时间下采样，所有整数层有效长度均为原始 lengths。
- LayerNorm 仅跨通道，不跨时间或样本，避免 batch/Padding 长度影响有效位置。
- 输入及每个块的无效位置置零，整数节点内部也掩码，避免 affine 偏移产生伪事件。
- 用 masked_fill 移除 Padding，即使 Padding 含 NaN 也不应污染有效输入。
- lengths 与 mask 同时提供时必须一致；旧模型的下采样 mask 不可用于本模型。
- 空签名的嵌入与分类结果为零，不应作为有效训练数据。
- 没有膜电位跨调用泄漏；当前 API 是整段推理，不提供卷积缓存式流式接口。
- 所有计数仍储存在浮点张量中，便于自动求导；不表示 GPU 自动使用整数算术。

## 5. 接入与实验隔离

- 模型名：`integer-tcn`，独立于 `snn` 和 `cnn`。
- 模型：`src/network_integer_tcn.py`。
- 配置：`configs/integer_tcn.json`。
- 集中式入口：`src/train_integer_tcn.py`；复用 `main.py` 的 AP+CE 训练，
  数据加载独立为 `real_forgery_data.py` + `dataset/datasetRealForgery.py`。
- 不修改 `formal_method.json`、旧神经元、既有模型或实验结果。
- 配置入口要求显式 run-tag；已存在的同 tag/seed 检查点目录拒绝覆盖。
- 模型 state_dict 的 `_extra_state` 保存 D、扩张率、宽度等。严格加载时校验，
  评估根据 metadata 自动重建，不受评估端默认 D=4 等参数影响。
- 复用 `evaluate_stylus.py` 和 embedding-only verifier；也适配全分辨率整数计数
  描述子与序列特征导出，但主实验先采用 embedding-only。
- 此次不扩展正式联邦基线矩阵。旧联邦入口仍训练原来的 FedBurst 编码器；
  禁止直接把含 `_extra_state` 的新检查点交给旧的纯张量聚合代码。

训练目标保持原有 DLM AP + label-smoothed CE，不使用 tau 方差奖励。
每个 episode 为 anchor + 同一身份的真实签名 positives + 针对该身份的真实熟练伪造 negatives，
同组采样不放回。默认 4 个不同身份，每身份 1+5+10 条，共 64 条。
不生成伪造，不调用 SLN，不读取合成缓存；不足样本直接报错，不合成或用其他身份样本凑数。
协议改为 real-skilled-forgery-supervised，不得继续称为 forgery-free。
默认 AdamW，lr=1e-3，5 epoch warm-up + cosine，100 epochs，梯度裁剪=1。
卷积/线性权重 decay=1e-4；Norm、偏置、残差缩放不做 weight decay。
这些是初始设置，而非已调优结论。

## 6. 使用方法（保留的 MSDS 预设）

当前默认 DeepSignDB 的命令见 [新实验文档](DEEPSIGNDB_CENTRAL_BASELINES.md)。
以下是后续 MSDS 使用方式，不是当前默认实验。

以下命令的工作目录均为 `fedburst_sig`，Python 使用当前项目的依赖环境。
模型训练入口不会在导入时执行训练，`--help` 不加载数据。

新目录准备命令（默认目标已准备好时不重复执行）：

```powershell
python src/prepare_real_forgery.py
```

训练（显式选择保留的 MSDS 配置，不需要 SLN 参数）：

```powershell
python src/train_integer_tcn.py --config configs/integer_tcn_msds.json --run-tag integer-tcn-msds-real-d4-v1 --seed 1111
```

修改 D 或时间上下文：

```powershell
python src/train_integer_tcn.py --config configs/integer_tcn_msds.json --run-tag integer-tcn-msds-real-d2-v1 --seed 1111 --integer-max 2 --tcn-dilations 1,2,4,8
```

配置文件可通过 `--config <path>` 替换；显式 CLI 参数优先于配置默认值。
新入口要求 `training_source=real`；指向旧 `data/` 或没有真实数据 manifest 的目录会报错。
直接调用 `main.py --model-name integer-tcn` 时，也默认真实来源，但必须显式指定
`--train-datasets msds --data-root data_real_forgery/msds_v1`。
旧 `cnn/snn` 入口仍默认原 synthetic/SLN 流程，原脚本及原配置不变。
`--tcn-embed-dim` 用于新模型；旧的 `--embed-dim`、`--sim-steps`、
`--spike-gradient`、`--surrogate-sigma` 不控制新模型的维度/单步/梯度定义。
启用旧 Burst/PLIF/复位分支等不兼容开关会报错。

导出普通验证嵌入（架构自动从检查点恢复）：

```powershell
python src/evaluate_stylus.py --model-name integer-tcn --checkpoint models/integer-tcn-msds-real-d4-v1/1111/epochEnd --run-tag integer-tcn-msds-real-d4-v1 --seed 1111 --eval-datasets msds --data-source real --data-root data_real_forgery/msds_v1
python src/verifier_stylus_embedding.py --run-tag integer-tcn-msds-real-d4-v1 --seeds 1111 --datasets msds --mode sf --embedding-normalization l2
python src/verifier_stylus_embedding.py --run-tag integer-tcn-msds-real-d4-v1 --seeds 1111 --datasets msds --mode rf --embedding-normalization l2
```

验证集导出需追加 `--eval-split validation --export-tag integer-tcn-msds-real-d4-v1-validation`，
以独立 tag 保存，避免与测试特征混用。`--data-source auto` 会根据新目录的 manifest 选用
同一套真实数据特征提取器。训练配置和导出文件均记录数据来源/划分/特征版本/校验值。

可选 `--export-seq-feats` / `--export-snn-tvs --export-all-layer-spike`。
整数描述子的平均值表示每采样点的平均事件计数，不是限制在 [0,1] 的二值发放率。

协议注意：现有 embedding-only verifier 复用旧 TVS 的模板/查询选取方式；它并未
自动读取 `configs/protocols/msds_cross_session_protocol.json`。上述 MSDS 命令用于
打通现有验证流程，不能把其结果直接标为严格 Session1→Session2 跨会话指标。
要报告该指标，后续须独立接入并测试该协议文件；本次不改动旧验证器的样本选取。

## 7. 验证与后续实验

```powershell
python -m pytest tests/test_real_forgery_data.py tests/test_integer_tcn.py tests/test_spike_gradient.py tests/test_federated_baselines.py -q
python src/benchmark_integer_tcn.py --device cpu --batch-size 2 --lengths 256,1024,4096
```

必须通过：整数前向/clipped STE、D 不引起梯度倍增、块内卷积输入整数、
Padding/混合长度/布局不变性、序列因果性、空短序列、反向更新、检查点配置校验、
实际特征导出，以及旧模型回归测试。

后续按同一划分与调参预算比较：原模型、同骨干 ANN、D=1/2/4/8，必要时加入教师蒸馏。
固定五种子 1111/2222/3333/4444/5555，先在验证集选方案，正式测试结果不参与调参。
报告 SF/RF EER、参数量、相同硬件及长度分布下的延迟/吞吐、静默/饱和情况。

边界：本次实现不包含同骨干 ANN、蒸馏、二值事件部署、从原始采集文件重建 MSDS 或正式训练。
烟雾测试和随机数据基准不构成 EER 改善、优于旧模型或真实节能的证据。

## 8. 本次实际验证记录

2026-09-10，Python 3.9.19，PyTorch 2.1.1+cu118：

- 新模型测试与两个现有回归测试文件合计 **46 passed**，耗时 12.07 秒。
- 唯一警告来自 PyTorch 的 TypedStorage 弃用提示，不是测试失败。
- 已用合成小样本执行 AP+CE 反向与参数更新。
- 已跑通实际训练入口保存 `epochEnd` 和配置，以及评估入口自动恢复 D/扩张率；
  测试数据加载器为替身，不涉及正式训练集。重复同 tag/seed 的输出被拒绝覆盖。
- 新评估入口绕过旧 FL 的 tensor-alias 展开，保留检查点 `_extra_state`。

CPU 前向基准：Intel64 Family 6 Model 183 Stepping 1，单线程，batch=2，
随机未训练输入，预热 3 次，测量 10 次，无梯度。默认模型、10 类训练头：
总参数 149,760；去掉分类头后 147,200。不同实际训练类别数会改变分类头参数量。

| 每条签名采样点数 | 每批平均耗时 | 最小～最大 |
|---|---:|---:|
| 256 | 5.95 ms | 5.54～6.59 ms |
| 1024 | 20.32 ms | 19.18～22.86 ms |
| 4096 | 101.73 ms | 99.91～105.05 ms |

三个长度的嵌入均为有限值。4096 点时八个整数节点的活跃比例约 26.7%～31.0%，
D=4 饱和值占比约 0.014%～0.190%。这是随机输入初始化状态，不代表真实签名训练后分布。
本次未测 GPU、反向耗时、与旧模型的相对加速比、真实 EER 或能耗。

## 9. 真实伪造数据切换的验证记录

2026-09-10 已将 MSDS 三个原始轨迹文件独立复制到 `data_real_forgery/msds_v1/`，
共 180,066,862 字节，另保存身份划分、跨会话协议和 manifest。逐文件核验源/副本 SHA-256，
复制完成及真实样本前向检查后再次确认旧文件哈希未变。

- 训练：225 身份，4500 真实签名 + 4500 真实熟练伪造。
- 验证：56 身份，1120 + 1120；测试：121 身份，2420 + 2420。
- 已核对实际文件身份与原 MSDS split 一致，三组身份互不重叠。
- 当前真实数据流程没有 SLN 参数读取、合成器导入或磁盘特征缓存写入。
- 用新目录 train split 的一个真实小组完成 CPU 前向：输入 `[3,935,12]`，
  类型顺序为 genuine/genuine/forgery，输出 `[3,256]`，特征与嵌入均有限。
- 真实数据新测试及旧回归合计 **61 passed**，15.53 秒；仅有 PyTorch TypedStorage 弃用警告。
- 测试覆盖独立副本、原文件不变、身份泄漏/篡改拒绝、真实 episode 顺序、无 SLN 导入、
  短序列与常数通道、训练/评估特征一致、实际入口训练保存和导出，以及旧模型默认路由不变。

实际数据仅进行了读取/特征提取/前向检查，未启动正式训练、调参或测试 EER 计算。
