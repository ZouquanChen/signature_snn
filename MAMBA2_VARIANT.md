# Integer-TCN + Mamba-2：独立长序列变体

日期：2026-09-10。模型名 `integer-tcn-mamba2`。独立配置为
`configs/integer_tcn_mamba2.json`，**没有替换原 Integer-TCN 默认配置，没有启动正式训练**。

## 1. 为什么加在这里

原 Integer-TCN 在最后一个时间位置的局部感受野为 67 点，随后进行均值/标准差池化。
加入 Mamba-2 式选择性状态空间模块，测试“先学习局部运动模式，再聚合更长上下文”的假设：

```text
12 维动态特征
  → 原始因果输入投影
  → 原始 4 个整数 TCN 残差块（D=4，扩张率 1/2/4/8）
  → 连续 Mamba-2 残差块 × 2
  → Masked mean + std
  → Linear 256 → 256 + L2
  → 签名嵌入 / 528 类训练分类头
```

两层新增模块宽度 128，expand=1，head_dim=32（4 heads），state_dim=16，
短深度卷积核 4，SSD 分块长度 64，残差缩放初始化 0.1。
这些是适合小模型和 8 GB 显存的起始设置，不是经过 EER 调优的最优值。

原来的八个整数发放节点完全保留；**Mamba 部分的投影、门控、状态与归一化都是连续值**。
不能把它报告为全脉冲 Mamba、全整数算术或直接获得事件硬件节能的网络。
它也没有解决显式局部序列对齐：仍使用全局嵌入评分，没有加入 DsDTW 的 DTW 匹配。

## 2. 采用的 Mamba 机制

参考 [Mamba-2 论文](https://arxiv.org/abs/2405.21060) 与作者的
[Mamba2Simple](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba2_simple.py)、
[SSD 最小参考](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/ssd_minimal.py)。

每个位置从输入投影得到门控 z、值 x、状态写入 B、状态读出 C 和选择步长 dt。
x/B/C 经过一个因果短卷积与 SiLU；B/C 在四个 heads 之间共享（ngroups=1）。

离散状态方程为：

`h_t = exp(-exp(A_log) * dt_t) * h_(t-1) + (dt_t * x_t) outer B_t`

`y_t = h_t @ C_t + D_skip * x_t`

随后 `y * SiLU(z)` → RMSNorm → 输出投影，并与输入残差相加。
这里 `D_skip` 是 SSM 的跳连参数，与整数发放上限 D=4 无关。
dt 是输入驱动的选择参数，不应直接解释为物理采样间隔 10 ms。

因此数学上仍有真实的时间状态依赖，范围可以超出 TCN 的局部窗口；但对当前任务是否有
精度收益尚未验证，长状态也不保证自动解决速度变化、伪造细节或跨用户泛化。

## 3. 为什么能在当前环境执行

本机检查结果：Windows、Python 3.9、PyTorch 2.1.1+cu118，没有安装
mamba_ssm、Triton 或 causal_conv1d。作者当前仓库的核心要求列出 Linux/Python 3.10+。
本次没有升级 Python、安装新依赖或替换已有训练环境。

采用项目内的 **纯 PyTorch 移植后端 `torch-ssd-prefix-v1`**：

- 块内使用 SSD 因果矩阵分解，只有 `chunk_size × chunk_size` 的局部矩阵。
- 块间利用仿射状态转移的可结合性，用对数层数的并行前缀组合。
- 不构造完整 `L × L` 矩阵，不在 Python 中逐采样点递推。
- 尾部 Padding 的输入/读出为零、转移为单位，排除无效步；每次独立调用从零状态开始。
- 支持任意序列长度和 mask，不要求 L 整除 chunk_size。

该参考实现不是官方融合 CUDA/Triton 内核。固定通道/状态维度时，块内工作量与
`L * chunk_size` 成比例，跨块前缀额外有 `log(L/chunk_size)` 开销；**不能把官方线性工作量
内核的吞吐、内存和加速数字直接套用到本实现**。

同样，当前 API 不提供带卷积缓存的流式推理；全签名预处理和池化仍依赖整段输入。
保留作者版权声明，附 Apache-2.0 许可和改动说明于 `third_party/mamba/`。

## 4. 实验隔离和使用

使用同一个独立 DeepSignDB 数据目录，528 个真实训练身份、442 个评估身份；
不混入其他数据，不新建验证集，仍用最低 epoch 平均训练 AP+CE 保存权重。
loss、训练预算和评分器默认保持不变，Mamba 变体默认从头初始化，不自动加载旧 TCN 权重。
正式验证仍是 1vs1、4vs1，并分别报告与原 Integer-TCN 和外部 DsDTW 的差距。

本次没有开启训练。用户确认后可在 `fedburst_sig` 下执行：

```powershell
python src/run_integer_tcn_experiment.py --config configs/integer_tcn_mamba2.json --run-tag integer-tcn-mamba2-deepsign-loss-s1111-v1 --seeds 1111 --device cuda --microbatch-tasks 1 --cpu-threads 4
```

该入口会自动保存日志、best_train_loss/last/final 权重并执行训练后评估。
新名字确保不覆盖旧训练结果。所有 Mamba 构型与后端版本写入 checkpoint 元数据，
评估按元数据重建；Mamba 计算代码也纳入运行代码哈希。

手工导出使用 `--model-name integer-tcn-mamba2`；普通旧 TCN 仍使用 `--model-name integer-tcn`。
旧模型仍能独立构建及加载，旧权重和实验结果未修改。已完成运行的历史代码哈希不变，
不要拿更新后的代码冒充原运行的精确续跑；新试验须用新的 run-tag。

## 5. 验证和资源检查

测试涵盖：分块 SSD 与逐步状态方程的输出/梯度一致、非整块长度、初始状态、
chunk_size 不影响数学结果、因果性、Padding/空序列、超过 TCN 窗口的历史影响、
分块梯度累积与 activation checkpoint 等价、模型工厂/配置/检查点恢复，以及旧模型回归。
逐步状态方程仅作为小样本测试的 oracle，不进入生产前向。

GPU 检查使用随机输入、16 条/批、FP32、activation checkpoint；仅计算 CE 前向/反向，
不更新权重，不是完整 AP+CE 训练测速。RTX 4070 Laptop，PyTorch 2.1.1：

| 长度 | 峰值 allocated 显存 | 平均前向+反向耗时 |
|---:|---:|---:|
| 1024 | 286.74 MB | 41.18 ms |
| 4096 | 1089.30 MB | 285.17 ms |
| 11819 | 3188.66 MB | 897.19 ms |

预热 1 次、测量 2 次，样本少且受设备负载影响，仅证明当前环境可完成长序列反向，
不能据此预测完整 100 epochs 耗时、相对官方 Mamba 的速度或真实能耗。
参数量：10 类检查模型 259,928；换成正式 528 类训练头为 392,536；
去掉身份分类头的编码器为 257,368 参数。

尚无新 EER。原单种子 Integer-TCN 为 10.04896% / 6.01201%；
DsDTW 外部 baseline 为 4.04±0.04% / 2.54±0.11%。增加 Mamba 不代表已改善这些指标。

最终验证：**87 项测试通过**（19.99 秒），仅有 PyTorch TypedStorage 弃用提示。
还额外使用正式 528 类头，在 `[16,11819,12]` 的随机输入上计算完整 AP+CE 并反向：
loss 和所有已产生的梯度均有限，Mamba 的 log_A 参数梯度非零，峰值 allocated
显存为 3194.92 MB。该检查没有 optimizer.step，不是正式训练。
