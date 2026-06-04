# 离线 TTA 可靠性诊断管道使用文档

本文档说明如何使用当前代码框架完成 **ImageNet-C / CIFAR-10-C 在线 corruption 诊断实验**，并生成 per-batch 的 EATA / SAR / ActMAD / SPA 指标、优化轨迹与分析图表。

> 重要定位：本项目实现的是 **离线诊断工具**，不是在线 TTA 算法。诊断阶段允许使用 ground-truth label 来优化输入并比较 `δ_model` 与 `δ_data`，不代表测试时在线自适应协议。

## 1. 环境准备

### 1.1 Python 依赖

代码依赖 PyTorch、torchvision 与常见科学计算/可视化库。建议在具备 CUDA 的环境中安装：

```bash
pip install torch torchvision numpy pandas scipy scikit-image opencv-python pillow matplotlib seaborn
```

如果只做静态语法检查，可运行：

```bash
python -m compileall config.py data models probe analysis viz run_experiment.py
```

### 1.2 数据目录

默认数据根目录在 `ProbeConfig.data_root` 中设置为：

```text
/Dataset/yezhong
```

代码支持两类数据集：

| 数据集 | 用途 | 默认读取方式 |
| --- | --- | --- |
| `cifar10_c` | 快速调试 | 读取 `/Dataset/yezhong/cifar-10-batches-py`，在线生成 corruption |
| `imagenet_c` | 主实验 | 读取 ImageNet clean val，在线生成 corruption；当前服务器若未下载 ImageNet，可暂时跳过 |

ImageNet clean val 会按以下候选路径顺序查找：

```text
{data_root}/imagenet/val
{data_root}/ImageNet/val
{data_root}/ILSVRC2012/val
{data_root}/val
```

数据 loader 每个 batch 返回：

```python
(x_corrupted, x_clean, y)
```

其中 corruption 在 `__getitem__` 阶段在线生成，不会把 ImageNet-C / CIFAR-10-C 副本缓存到磁盘。

## 2. 代码结构速览

```text
config.py                 # ProbeConfig，集中管理实验参数
data/                    # 在线 corruption 数据层与 source stats 预计算
models/                  # 模型加载、冻结、ActMAD 层选择、activation hooks
probe/                   # 输入优化器与 per-batch 指标计算
analysis/                # 相关性、条件、轨迹、因果、λ2 有效性分析
viz/                     # 公用绘图保存函数
scripts/                 # 常用入口脚本
run_experiment.py         # 全流程主入口
```

核心配置类是 `ProbeConfig`，默认包含 corruption 列表、severity 列表、batch size、模型名、优化步数、`lambda1/lambda2`、频域 bin 数与输出目录等参数。

## 3. 快速开始：CIFAR-10-C 调试

CIFAR-10-C 调试是推荐的第一步，因为它不依赖 ImageNet 数据。

```bash
bash scripts/run_debug.sh
```

该脚本等价于运行：

```bash
python run_experiment.py \
  --dataset cifar10_c \
  --data-root /Dataset/yezhong \
  --output-dir ./outputs_debug \
  --batch-size 8 \
  --opt-steps 2 \
  --max-batches 1
```

调试脚本会：

1. 加载默认 `resnet50` 预训练模型并冻结参数；
2. 对每个 corruption/severity 的第一个 batch 在线生成 corrupted images；
3. 过滤误分类样本 `B_err`；
4. 对输入 `x` 运行少量 Adam 优化步；
5. 写出 `outputs_debug/metrics.csv` 和 `outputs_debug/traj_logs/*.pkl`；
6. 在 `outputs_debug/analysis/` 下生成相关性、条件分析和因果对照图。

> 注意：CIFAR-10 图像会 resize 到 224×224，以适配 torchvision ImageNet 预训练模型。它主要用于验证管道是否能跑通，不应作为主论文结果。

## 4. Source Statistics 预计算

ActMAD 需要 clean source domain 的 activation mean/variance；SPA 需要 clean source domain 的 `RAPSD_src`。请在主实验前运行预计算。

### 4.1 ImageNet source stats

```bash
bash scripts/run_precompute.sh
```

或显式指定参数：

```bash
python data/precompute_stats.py \
  --dataset imagenet_c \
  --data-root /Dataset/yezhong \
  --model-name resnet50 \
  --output ./outputs/source_stats \
  --batch-size 64
```

输出目录为：

```text
outputs/source_stats/{model_name}/
├── activation_stats.pkl
└── rapsd_src.npy
```

### 4.2 调试版 source stats

如果只想先跑通 CIFAR-10-C 调试版，可运行：

```bash
python data/precompute_stats.py \
  --dataset cifar10_c \
  --data-root /Dataset/yezhong \
  --model-name resnet50 \
  --output ./outputs_debug/source_stats \
  --batch-size 32 \
  --max-batches 5
```

然后在主流程中使用相同的 `source_stats_path`。如果没有找到 source stats，主流程会用空 activation stats 和零 RAPSD source curve 继续运行；这适合调试管道，但不适合正式实验。

## 5. 运行主实验

### 5.1 默认 ImageNet-C 主实验

```bash
bash scripts/run_main.sh
```

等价于：

```bash
python run_experiment.py \
  --dataset imagenet_c \
  --data-root /Dataset/yezhong \
  --output-dir ./outputs
```

### 5.2 指定模型、优化步数与 λ 参数

```bash
python run_experiment.py \
  --dataset imagenet_c \
  --data-root /Dataset/yezhong \
  --model-name resnet101 \
  --output-dir ./outputs_resnet101_l2_0 \
  --batch-size 64 \
  --opt-steps 100 \
  --lambda1 1.0 \
  --lambda2 0.0
```

### 5.3 Variant A/B/C 对照

实验指引中的三个变体可用以下命令运行：

```bash
# Variant A: 纯模型视角，主实验
python run_experiment.py --dataset imagenet_c --lambda1 1.0 --lambda2 0.0 --output-dir ./outputs_varA

# Variant B: 语义引导，有效性对照
python run_experiment.py --dataset imagenet_c --lambda1 1.0 --lambda2 0.5 --output-dir ./outputs_varB

# Variant C: 强引导，退化检测
python run_experiment.py --dataset imagenet_c --lambda1 1.0 --lambda2 2.0 --output-dir ./outputs_varC
```

运行完成后，可在 Python 中调用 `analysis.validity.analyze_lambda2_effect` 比较 Variant A 与 B：

```python
import pandas as pd
from analysis.validity import analyze_lambda2_effect

metrics_varA = pd.read_csv("./outputs_varA/metrics.csv")
metrics_varB = pd.read_csv("./outputs_varB/metrics.csv")
analyze_lambda2_effect(metrics_varA, metrics_varB, "./outputs_validity/analysis")
```

### 5.4 灵敏度分析

可用 shell loop 扫描 `lambda1 × opt_steps`：

```bash
for lambda1 in 0.5 1.0 2.0; do
  for steps in 50 100 200; do
    python run_experiment.py \
      --dataset imagenet_c \
      --model-name resnet50 \
      --lambda1 "$lambda1" \
      --lambda2 0.0 \
      --opt-steps "$steps" \
      --output-dir "./outputs_sensitivity/lambda1_${lambda1}_steps_${steps}"
  done
done
```

## 6. 输出文件说明

主流程会在 `output_dir` 下生成：

```text
outputs/
├── metrics.csv
├── traj_logs/
│   └── {corruption}_{severity}_{batch_idx}.pkl
└── analysis/
    ├── corr_global.png
    ├── corr_{corruption}.png
    ├── boxplots_by_family.png
    ├── severity_metric_means.png
    ├── rapsd_shift_mode_heatmap.png
    ├── causal_groupA_stats.json
    ├── causal_groupB_stats.json
    └── causal_comparison.png
```

`metrics.csv` 的列包括：

```text
batch_id, model_name, corruption, severity, lambda2,
H_err, H_star, delta_H,
G_err, G_star, delta_G,
A_err, A_star, delta_A,
low_freq_ratio, cosine_sim_delta, l2_ratio_delta
```

其中：

| 列 | 含义 |
| --- | --- |
| `H_err/H_star/delta_H` | EATA batch mean entropy before/after probe |
| `G_err/G_star/delta_G` | SAR batch mean entropy 对模型参数梯度的 L2 norm before/after probe |
| `A_err/A_star/delta_A` | ActMAD activation mean + variance 偏差 before/after probe |
| `low_freq_ratio` | `δ_model` RAPSD 的低频能量占比 |
| `cosine_sim_delta` | `δ_model` 与 `δ_data` 的平均 cosine similarity |
| `l2_ratio_delta` | `||δ_model||₂ / ||δ_data||₂` 平均比值 |

## 7. 断点续跑机制

`run_experiment.py` 会检查输出目录中的 `metrics.csv`。如果某个 `batch_id` 已存在，则跳过该 batch。

`batch_id` 格式为：

```text
{corruption}_{severity}_{batch_idx}
```

因此，如果实验中断，只需用相同参数重新运行同一命令即可继续。若改变 `model_name`、`lambda1`、`lambda2`、`opt_steps` 等关键设置，建议使用新的 `output_dir`，避免把不同实验配置混写到同一个 CSV。

## 8. 关键实现细节与注意事项

### 8.1 模型参数不会被优化

`load_frozen_model` 会把所有模型参数的 `requires_grad` 设为 `False`。`run_probe` 优化器只接收输入张量 `x`，因此 Adam 更新只作用于输入。

### 8.2 SAR 的 `G_bar` 是 batch 级定义

`compute_G_bar` 会临时打开模型参数梯度，对整个 batch 的 mean entropy `H̄(B)` 调用 backward，然后统计所有模型参数梯度的 L2 norm。这不是逐样本梯度范数的均值。

### 8.3 ActMAD 小 batch warning

ActMAD 使用 batch activation variance。若 batch size 小于 `config.min_batch_size_for_actmad`，方差估计不稳定，代码会记录 warning。正式实验建议使用 `batch_size >= 32` 或默认 `64`。

### 8.4 `save_xstar` 默认关闭

`ProbeConfig.save_xstar=False` 时不会持久保存完整 `x*` tensor，以减少磁盘占用。指标计算仍可通过 `x_orig + delta_model` 重建 `x_star`。

### 8.5 ImageNet 暂未下载时的推荐流程

如果 `/Dataset/yezhong` 下暂时只有 CIFAR-10：

1. 先运行 `bash scripts/run_debug.sh` 验证管道；
2. 等 ImageNet val 准备完成后运行 `scripts/run_precompute.sh`；
3. 再运行 `scripts/run_main.sh` 或按消融矩阵自定义命令。

## 9. 推荐实验执行顺序

1. **环境检查**：确认 `torch`、`torchvision`、`numpy`、`pandas`、`matplotlib`、`seaborn` 等可 import。
2. **CIFAR-10-C debug**：运行 `bash scripts/run_debug.sh`，确认 `metrics.csv` 与图表生成。
3. **source stats 预计算**：对目标模型运行 `data/precompute_stats.py`。
4. **主实验**：对 `resnet50/resnet101/vit_b16` 分别运行 `lambda2=0` 主实验。
5. **有效性对照**：对 `resnet50` 运行 `lambda2=0.5`，调用 `analyze_lambda2_effect`。
6. **灵敏度分析**：扫描 `lambda1 ∈ {0.5,1.0,2.0}` 与 `opt_steps ∈ {50,100,200}`。
7. **论文归纳**：优先报告跨架构一致结论；单架构结果只作为补充讨论。

## 10. 常见问题

### Q1: `FileNotFoundError: ImageNet clean validation directory not found`

说明当前 `data_root` 下没有可识别的 ImageNet val 目录。请改用 `--dataset cifar10_c` 做调试，或把 ImageNet val 放到文档第 1.2 节列出的候选路径之一。

### Q2: 为什么 `A_err/A_star` 是 NaN？

通常是因为没有加载 source activation stats，或 `source_stats_path/{model_name}/activation_stats.pkl` 不存在。正式实验前请先运行 source stats 预计算。

### Q3: 为什么运行很慢？

每个 batch 都会执行输入优化，并且每步还要记录 H/G/A/SPA 轨迹；其中 SAR 的 `G_bar` 需要对模型参数做 backward，ActMAD 需要 hook forward。调试时请减小 `--batch-size`、`--opt-steps`、`--max-batches`。

### Q4: 可以修改 corruption 列表吗？

可以。在 Python 中实例化 `ProbeConfig` 后修改 `config.corruption_families` 和 `config.severities`，再调用 `run_full_pipeline(config)`。

```python
from config import ProbeConfig
from run_experiment import run_full_pipeline

config = ProbeConfig(dataset="cifar10_c", data_root="/Dataset/yezhong", output_dir="./outputs_subset")
config.corruption_families = ["gaussian_noise", "brightness"]
config.severities = [1, 3, 5]
config.max_batches = 2
run_full_pipeline(config)
```
