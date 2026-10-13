# DeepSignDB 集中式真实伪造实验与 ANN baseline

资料核对日期：2026-09-10。本文的 ANN 数值是外部报告结果，当前正式训练进度见运行记录。

后续运行更新：用户已确认不划验证集，按最低 epoch 平均训练 AP+CE 选择权重，
并只先跑种子 1111。正式运行已启动，详见 [实验运行记录](EXPERIMENT_RUNS.md)。
下文原先“固定最终检查点”的选择约定由这次用户确认覆盖；数据文件和身份范围未改变。

## 1. 当前实验固定条件

第一阶段采用 **DeepSignDB / stylus / 集中式 / 真实熟练伪造监督**。
不混用 MSDS、GPDS、finger 数据，不加入联邦训练、合成伪造、HN 或 TVS 融合。

选择与 DsDTW 作者公开训练代码一致的 528 个开发身份，而不是声称使用了全部 1084 个：

| 开发/训练子集 | 身份 | 真实签名 | 真实熟练伪造 |
|---|---:|---:|---:|
| MCYT | 230 | 5750 | 5750 |
| BiosecurID | 268 | 4288 | 3216 |
| e-BioSign DS1 | 30 | 1200 | 900 |
| 合计 | 528 | 11238 | 9866 |

评估包含全部 442 个 stylus 评估身份：MCYT 100、BiosecurID 132、Biosecure DS2 140、
e-BioSign DS1 35、e-BioSign DS2 35。e-BioSign DS1 按五个设备分场景评估，不能把
设备场景数误称为额外身份数。当前没有 finger 输入，也没有用到 EBio2_dev 的 46 个身份。

训练每批 4 个不同身份，每身份 anchor 1 + genuine 5 + skilled forgery 10，共 64 条，
与 DsDTW 公开默认 episode 一致。12 维动态特征的定义、15 Hz/100 Hz 滤波及逐签名
标准化与作者代码对应；本实现额外保护短序列和常数通道，保持原轨迹不被原地修改。

依据：[DsDTW 作者训练入口](https://github.com/KAKAFEI123/DsDTW/blob/main/main_DsDTW.py)、
[真实伪造采样](https://github.com/KAKAFEI123/DsDTW/blob/main/dataset/datasetTrainAll_SF.py)、
[特征定义](https://github.com/KAKAFEI123/DsDTW/blob/main/dataset/utils.py)。

## 2. 身份编号与数据安全

DeepSignDB 的 Development 与 Evaluation **分别编号**。
上游预处理脚本注明开发 MCYT 为 1–230，评估 MCYT 为 1–100；BiosecurID 也跨 split
复用部分数字。因此，上一轮看到的数字重叠不能直接证明同一个人泄漏。
官方声明原始两部分包含不同身份；这里按 `(official split, corpus, ID)` 理解身份。

核对证据：本地 `D:/SynSig2Vec-main/preprocess/DeepSignDB_dev.py` 与
`DeepSignDB_eva.py` 的编号说明；[官方数据说明](https://github.com/BiDAlab/DeepSignDB)
明确 Development/Evaluation 的不同身份划分。

本次还核对了本地文件的全部预期 ID 范围、每人样本数，以及跨训练/评估的精确轨迹哈希，
未发现相同轨迹。零重复不是独立的生物身份真实性证明；原始采集身份映射没有在本轮重建。
同一 split 内的原有重复样本没有擅自清洗，以免改变已发布方法的数据条件。

新目录：`data_real_forgery/deepsigndb_stylus_v1/`，包括八个独立轨迹副本、
`manifest.json` 和 `protocol.json`。旧 `data/`、MSDS 副本、合成缓存和旧训练结果均保留。
文件逐一校验 SHA-256；已存在的目标目录拒绝覆盖。

## 3. 可引用的 ANN 指标

以下是**作者已报告的外部 benchmark 结果，不是本项目复现结果**；指标为 EER (%)，越低越好。
1vs1/4vs1 表示每位评估身份用 1/4 条真实参考签名进行验证，不是模型训练只用 1/4 条样本。

| ANN 方法 | Stylus SF 1vs1 | Stylus SF 4vs1 | Stylus RF 1vs1 | Stylus RF 4vs1 | 适用定位 |
|---|---:|---:|---:|---:|---|
| RNN，DeepSign 2021 | 8.5 | 7.9 | 未在对应表中报告 | 未在对应表中报告 | 官方历史 ANN 参考 |
| TA-RNN，DeepSign 2021 | 4.2 | 3.3 | 1.5 | 0.6 | 官方核心 ANN benchmark |
| DsDTW，TIFS 2022 | 4.04 ± 0.04 | 2.54 ± 0.11 | 1.69 ± 0.15 | 0.97 ± 0.08 | 本实验最接近的已发布 ANN baseline |

RNN/TA-RNN 数据直接核对 [DeepSign 原论文 Table V](https://arxiv.org/html/2002.10119v3)。
DsDTW 数据核对 [作者仓库的 Table VIII](https://raw.githubusercontent.com/KAKAFEI123/DsDTW/main/images/Results_on_DeepSignDB.jpg)，
出版 DOI：[10.1109/TIFS.2022.3180219](https://doi.org/10.1109/TIFS.2022.3180219)。
± 数值按作者表格保留，不把它解释成置信区间；本轮未重新训练或重算其方差。
作者公开验证脚本使用 111/222/333/444/555 五个种子。

注意：DsDTW 的 RF 并不优于 TA-RNN，不能只凭模型较新就宣称所有场景都更好。
SynSig2Vec 的 forgery-free 成绩不放在“同真实伪造监督条件”表中；DTW 不是 ANN。

## 4. 哪些条件一致，哪些仍不一致？

相对于 DsDTW 作者代码，已经对齐：

- 集中式学习，stylus 输入；三个开发子集及 528 个身份的范围。
- 使用训练身份的真实熟练伪造，而不是 Sigma–Lognormal 合成伪造。
- 12 维动态特征定义、每批 4 × (1+5+10) 的 episode 结构。
- 覆盖五个评估子集、442 个评估身份，固定参考，1vs1 四轮 / 4vs1 一轮。

仍有差异，不能称为严格受控复现：

- DsDTW 学局部序列并用 soft-DTW/DTW 及 triplet 类目标；Integer-TCN 是全局嵌入、AP+CE。
- DsDTW 默认 SGD、lr=0.01、20 epochs、指数衰减；本项目配置为 AdamW、lr=0.001、
  100 epochs、warm-up + cosine。训练预算、优化器不是相同条件。
- 作者种子编号不同；尚未验证原始预处理文件逐字节等同，也未复现作者权重。
- 新编码器没有 Sig2Vec 的 32 维多头结构，新集中式 verifier 用整个向量 L2，
  不能沿用旧 verifier 的 head-wise normalization。后端距离和校准方式也不完全相同。
- 本地没有官方 Comparison_Files，当前评估按公开脚本的样本选取规则重建；
  不能声称逐对比较文件已经完全一致。
- TA-RNN 论文使用官方开发集训练/验证划分；其开发数据范围与 528 人子集不完全相同。

结论：**DsDTW 可作为最接近的数据/监督/任务条件下的外部 ANN baseline；TA-RNN 是官方参考。
严格检验“脉冲化损失多少精度”，仍需用同一骨干的 ANN 在本地相同数据、预算、损失和
评分器下重训，不能直接拿以上出版数字充当该受控消融。**

## 5. 集中式评估协议

优先报告熟练伪造（SF）1vs1 与 4vs1。参考只来自前四条 genuine；1vs1 逐一选取，
4vs1 同时使用四条。查询规则由公开 DeepSign/DsDTW 协议对应：

- MCYT：21 条 genuine query、25 条 forgery query。
- BiosecurID：12 条 genuine query、12 条 forgery query。
- Biosecure DS2：只取后 15 条 genuine query、20 条 forgery query；不能把 30 条 genuine 全部作为 query。
- e-BioSign DS1：每设备 4 条 genuine query、6 条 forgery query。
- e-BioSign DS2：4 条 genuine query、6 条 forgery query。

出处：[DeepSign Table IV](https://arxiv.org/html/2002.10119v3)。
RF 表格指标保留作参考，但新的严格集中式结果入口当前只报告 SF；RF 的逐设备负样本选择
及官方 pair-list 尚未逐对核验，不能直接把旧 RF 入口的输出标为严格同条件结果。

新入口每次只评一个训练种子，然后报告五个种子的均值与样本标准差。
这与“先合并五个种子的 ROC，再得到一个 EER”不是同一个统计量。
论文 EER 的阈值扫描用于指标计算，不等于可在部署时用测试标签选阈值。

此 528 人作者配置没有独立 validation 文件。第一轮固定模型、预算和最终检查点，不按
evaluation EER 选择 epoch、D 或超参数。若要调参，需要先从 Development 划出验证身份，
对所有本地 ANN/SNN 使用同一划分，并注明训练身份数已变化。

## 6. 命令

工作目录为 `fedburst_sig`。新目录本轮已经准备好，不重复执行准备命令。

```powershell
python src/prepare_deepsigndb_real.py
python src/train_integer_tcn.py --run-tag integer-tcn-deepsign-central-real-d4-v1 --seed 1111
```

默认配置 `configs/integer_tcn.json` 已切换至 DeepSignDB，未实际启动上述训练。
不需要 SLN 参数。之后重复种子 2222/3333/4444/5555，保持同一个 run-tag。

```powershell
python src/evaluate_stylus.py --model-name integer-tcn --data-source real --data-root data_real_forgery/deepsigndb_stylus_v1 --eval-datasets deepsigndb --run-tag integer-tcn-deepsign-central-real-d4-v1 --seed 1111
python src/verifier_deepsign_central.py --run-tag integer-tcn-deepsign-central-real-d4-v1 --references 1
python src/verifier_deepsign_central.py --run-tag integer-tcn-deepsign-central-real-d4-v1 --references 4
```

五种子的特征都导出后再执行上述默认五种子验证；单种子烟雾检查可显式加 `--seeds 1111`。
可用 `--output <new_file.json>` 保存新结果文件，已有文件拒绝覆盖。

后续恢复 MSDS：`python src/train_integer_tcn.py --config configs/integer_tcn_msds.json --run-tag <new_tag>`。
本次不启动其他数据集或联邦实验。

## 7. 本次实现验证

- 默认配置已切至 DeepSignDB 集中式；MSDS 独立预设与副本保留。
- 八个实际数据文件已独立复制，SHA-256 与原文件一致。
- 预期 ID 范围/每人样本数校验通过，精确轨迹跨 train/evaluation 重复为 0。
- 从三个真实训练子集各取一个小组，完成 CPU 前向：`[9,1063,12] → [9,256]`，输出有限。
- 前向后再次核对旧 DeepSignDB 文件和 MSDS 副本，哈希均未变。
- 新数据、1/4 参考分支、逐种子统计、归一化及旧模型回归合计 **66 passed**，16.50 秒。
  唯一警告为 PyTorch TypedStorage 弃用提示。
- 尚未训练正式模型，未产生新 ANN/SNN 实验 EER；第 3 节数值仍全部是外部作者报告。
