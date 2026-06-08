# 离线二维 Gamma 结构诊断管道使用文档

本项目当前版本对应研究计划 v3.1：**Wavelet-grounded × Hierarchical Aggregation Framework**。主线目标是离线测量 `Gamma_c(k,l)`，分析 corruption/domain shift 在物理粗粒化尺度 `k` 与数据组织层级 `l` 上的残余偏移，并进一步解释冻结模型响应与 TTA prior signals。

本阶段不实现、不评估 online TTA stream、在线 buffer、在线样本选择、部署式模型更新或在线自适应协议。旧的 `x*` 输入优化仅保留为 supplement。

## 1. 核心定义

数据侧比较对象固定为同一批 test samples 的 aligned pair：

```text
P_0 = clean reference distribution
Q_c = corrupted/domain-shifted distribution
Gamma_c(k,l) = D(A_l(R_k(Q_c)), A_l(R_k(P_0)))
```

其中：

- `R_k`：DWT/IDWT coarse-graining，默认 `db4`，`J=3`。
- `A_l`：`pixel / patch / sample / label` 四个组织层级 descriptor。
- `D`：默认 RBF-MMD，可选 sliced Wasserstein 或 energy distance。
- `epsilon_l`：clean-clean descriptor bootstrap 的 95% 分位数。
- `k*_{data}(l)`：固定 level 后第一个进入 invariant basin 的尺度。

## 运行依赖

核心运行依赖：

```bash
pip install torch torchvision numpy pandas scipy scikit-image opencv-python pillow tqdm
```

绘图依赖可选：

```bash
pip install matplotlib seaborn pyparsing
```

若绘图依赖缺失，程序仍会写出 CSV/JSON 主结果，只跳过 heatmap。

## 2. 推荐执行顺序

### Phase 1/2：数据结构测量

```bash
python run_experiment.py \
  --phase structure \
  --dataset cifar10_c \
  --data-root /Dataset/yezhong \
  --output-dir ./outputs
```

调试烟雾测试：

```bash
bash scripts/run_debug.sh
```

输出：

```text
outputs/gamma/gamma.csv
outputs/gamma/gamma_summary.csv
outputs/gamma/gamma_summary.json
outputs/gamma/heatmaps/
```

`gamma.csv` 每行对应一个 `(corruption, severity, k, level)`：

```text
corruption, severity, k, level, gamma, epsilon, in_basin, distance
```

`gamma_summary.csv` 输出：

```text
k_star_pixel / patch / sample / label
not_in_basin_pixel / patch / sample / label
invariant_pairs
basin_area
basin_group
```

### Phase 3：冻结模型响应曲线

```bash
python run_experiment.py \
  --phase response \
  --dataset imagenet_c \
  --model-name resnet50 \
  --data-root /Dataset/yezhong \
  --source-stats-path ./outputs/source_stats \
  --output-dir ./outputs
```

输出：

```text
outputs/response/response_curves.csv
outputs/response/response_summary.csv
```

记录：

```text
Acc(k), H_bar(k), G(k), A(k), R_DWT(k)
k_best_model, Gain_coarse
```

### Phase 6：x* supplement

```bash
python run_experiment.py \
  --phase xstar \
  --dataset cifar10_c \
  --data-root /Dataset/yezhong \
  --output-dir ./outputs
```

输出位于：

```text
outputs/xstar_supplement/
```

该结果只用于比较 `W(delta_model)` 与 `Gamma_c(k,pixel/patch)` 的一致性，不进入主线结论。

## 3. 数据对齐原则

`P_0` 与 `Q_c` 必须由同一 sample index 配对得到。代码优先读取官方 corruption 数据：

- `CIFAR-10-C/<corruption>.npy`
- `ImageNet-C/<corruption>/<severity>/...`

如果 `--corruption-source auto` 找不到官方数据，会使用 deterministic synthetic corruption fallback，主要用于小规模调试。正式实验建议使用：

```bash
--corruption-source official
```

## 4. 常用参数

```bash
--corruptions gaussian_noise,brightness
--severities 1,3,5
--dwt-wavelet db4
--dwt-levels 3
--aggregation-levels pixel,patch,sample,label
--distance mmd
--epsilon-bootstrap 20
--epsilon-quantile 0.95
--max-descriptor-items 4096
--max-batches 10
```

## 5. 声明边界

本代码当前只支持：

```text
offline theoretical diagnosis / offline empirical evidence
```

不支持声明：

```text
online TTA algorithm performance
online stream robustness
test-time buffer routing effectiveness
deployed adaptation improvement
```
