# WEV4/GAC 免训练诊断结果分析

> 数据：
> - 1000：`CC/WEV4/logs/gac_diag_1000_full/`
> - 1000-GD：`CC/WEV4/logs/gac_diag_gd_full/`
> - 训练/测试原始日志：`CC/WEV4/logs/GAC_1000/`、`CC/WEV4/logs/GAC_GD_test/`
> 口径：seed1234 单 seed；同一 checkpoint（epoch 120）；full checkpoint bank；same-camera 排除；pre-BN CLS + L2。
> 标记：[代码事实] / [实验支持] / [机制假设] / [统计边界]。
> 注意：`gac_grad.json` 是 one-batch（32 PKM 样本=96 图）last-block+norm 梯度 proxy；head/rank/bank 是全量。

---

## 0. 一句话

**诊断结果把 V4 的机制问题钉死了：GAC 的三个 head 并没有形成三个 gallery-specific 方向，而是收敛成“两个近重复的通用身份分类器 + 一个偏 IR 的残差方向”；TIR head 甚至不是 TIR 查询的最佳 head，best scene 是 IR。GAC 的有效梯度只有 3–5%，且与 GPD/PCA 部分同向；bank 仍是 V2 的 class-mean 几何（`cos(P_id,mean P_scene)=0.9993`，TIR own_ratio=1.0）。因此 V4 的 R1/同模态收益是真实的，但 cross/TIR 的尾部和 mAP 不可能靠这组 heads 解决。**

---

## 1. 诊断本身可信吗：与官方日志对齐

诊断脚本的 rank audit 与官方 test log 完全对齐，说明后续新指标（R5/R10、pair bins）可信：

| 口径 | 诊断 | 官方 V4 | 结论 |
|---|---:|---:|---|
| 1000 mixed R1/R5/R10 | 30.80/47.98/55.93 | 30.80/47.98/55.93 | 完全一致 |
| GD mixed R1/R5/R10 | 31.20/45.93/53.86 | 31.20/45.93/53.86 | 完全一致 |
| 1000 九对平均 R1 | 19.97 | 19.97 | 完全一致 |
| 九对逐对 R1 | 43.14/27.92/7.73/25.71/29.56/6.93/8.29/8.43/22.01 | 同左 | 完全一致 |
| AG A→G R1/R10 | 35.26/60.07 | 35.26/60.07 | 完全一致 |
| AG G→A R1/R10 | 31.83/57.97 | 31.83/57.97 | 完全一致 |

- `meta.json`：checkpoint epoch=120、`bank_source=checkpoint`、500 类、1000 query 2135/模态、gallery 31203/模态；GD query 399/模态、gallery 6836/模态；GD 984 valid。
- `train_images=92133`（3×30711）；head CE/acc 是完整 train split。
- 因此：**head/bank/rank 的诊断不是近似；只有 one-batch gradient proxy 是采样估计。**

---

## 2. 全局指标：V4 的“强”和“弱”分别在哪

### 2.1 1000 / GD / AG 总表

| 协议 | R1 | R5 | R10 | >10 占比 |
|---|---:|---:|---:|---:|
| 1000 mixed | 30.80 | 47.98 | 55.93 | 44.08% |
| 1000 same 3 对 | 31.57 | 48.63 | 57.06 | 42.94% |
| 1000 cross 6 对 | 14.17 | 26.03 | 33.16 | **66.84%** |
| 1000 九对平均 | 19.97 | 33.56 | 41.13 | 58.87% |
| GD mixed | 31.20 | 45.93 | 53.86 | 46.14% |
| GD same 3 对 | 31.71 | 45.93 | 54.88 | 45.12% |
| GD cross 6 对 | 13.87 | 24.24 | 30.44 | **69.56%** |
| GD 九对平均 | 19.82 | 31.47 | 38.58 | 61.42% |
| AG A→G | 35.26 | 52.44 | 60.07 | 39.93% |
| AG G→A | 31.83 | 49.53 | 57.97 | 42.03% |

**直接读法：**

- 1000 的 R1/R5/R10 与官方一致；**cross 的 R5/R10 只有 26.03/33.16**，>10 占 **66.84%**。
- same 与 cross 的差距随 rank 扩大：R1 差 17.40pp、R5 差 22.60pp、R10 差 23.90pp，>10 差 23.90pp。**不是单纯 top-1 边界问题，而是 cross 的覆盖率/表征问题。**
- GD 的 cross >10 更高（69.56%）；GD 的整体 R1 31.20 比 1000 高 0.40，但 R5/R10 低 2.05/2.07，是 **top1/tail 的 trade**。相对 V2 的官方 GD 仍是 R1 −0.41、R5 +1.11、R10 −0.31，不能把 GD 简单说成全面退化。
- AG 的 R1/R10 已过 paper+1；新补的 R5 为 52.44/49.53，主要被 Thermal 拖累（见 §5.3）。

### 2.2 每个 query 模态的 mixed 指标

| 数据 | 模态 | R1 | R5 | R10 | >10 |
|---|---|---:|---:|---:|---:|
| 1000 | RGB | 40.56 | 59.86 | 66.70 | 33.30% |
| 1000 | NIR/IR | 30.12 | 47.17 | 55.64 | 44.36% |
| 1000 | TIR | 21.73 | 36.91 | 45.43 | 54.57% |
| GD | RGB | 43.60 | 60.37 | 66.77 | 33.23% |
| GD | NIR/IR | 31.10 | 44.82 | 53.35 | 46.65% |
| GD | TIR | 18.90 | 32.62 | 41.46 | 58.54% |

- 模态强弱顺序在所有口径下都是 **RGB > NIR > TIR**。
- TIR 在 1000 mixed 里 R1 只有 21.73、R10 45.43、>10 54.57%；GD 更差（18.90/41.46/58.54%）。
- 这不是“cross 才弱”：TIR 的 same 也弱（TIR→TIR R10 46.79，>10 53.2%）。

---

## 3. 九对细粒度：cross 尾部是主缺口

### 3.1 1000 九对（R1/R5/R10，新补 R5/R10）

| Query→Gallery | R1 | R5 | R10 | >10 |
|---|---:|---:|---:|---:|
| RGB→RGB | 43.14 | 61.12 | 68.43 | 31.6% |
| RGB→NIR | 27.92 | 44.59 | 52.69 | 47.3% |
| RGB→TIR | 7.73 | 17.80 | 24.31 | **75.7%** |
| NIR→RGB | 25.71 | 41.12 | 49.98 | 50.0% |
| NIR→NIR | 29.56 | 47.17 | 55.97 | 44.0% |
| NIR→TIR | 6.93 | 16.96 | 23.56 | **76.4%** |
| TIR→RGB | 8.29 | 18.45 | 24.78 | **75.2%** |
| TIR→NIR | 8.43 | 17.24 | 23.65 | **76.3%** |
| TIR→TIR | 22.01 | 37.61 | 46.79 | 53.2% |

### 3.2 GD 九对

| Query→Gallery | R1 | R5 | R10 | >10 |
|---|---:|---:|---:|---:|
| RGB→RGB | 44.82 | 62.50 | 70.12 | 29.9% |
| RGB→NIR | 28.66 | 39.33 | 45.12 | 54.9% |
| RGB→TIR | 8.84 | 19.82 | 27.13 | 72.9% |
| NIR→RGB | 24.39 | 38.41 | 44.21 | 55.8% |
| NIR→NIR | 30.79 | 43.90 | 52.13 | 47.9% |
| NIR→TIR | 6.10 | 15.24 | 21.95 | **78.0%** |
| TIR→RGB | 7.62 | 17.68 | 22.87 | **77.1%** |
| TIR→NIR | 7.62 | 14.94 | 21.34 | **78.7%** |
| TIR→TIR | 19.51 | 31.40 | 42.38 | 57.6% |

**读法：**

- TIR 参与的 6 个 cross 对（RGB→TIR、NIR→TIR、TIR→RGB、TIR→NIR）R5 只有 **14.94–19.82**、R10 **21.34–27.13**、>10 **72.9–78.7%**。
- RGB↔NIR cross 的 R5 约 **38–45**、R10 **44–53**；离 same 也有差距，但远好于 TIR cross。
- TIR→TIR 的 R1（22.01/19.51）看似“过 paper+1 R1”，但 R5/R10 只有 37.61/46.79（1000），>10 53.2%；**TIR same 的 tail 也很差**。
- NIR→NIR 的 R5/R10（47.17/55.97）反而明显好于 TIR→TIR；所以“只有 RGB/TIR”不是“NIR 更差”，而是 **TIR 的 paper+1 R1 门槛更低**（见 §7）。

---

## 4. Head 机制：没有形成 gallery-specific 方向

### 4.1 per-head train CE / acc（30711/模态，完整 train split）

| q \ head | RGB head | NIR head | TIR head | 最佳 head |
|---|---:|---:|---:|---|
| RGB | **.0901 / .9982** | .1201 / .9971 | .1532 / .9948 | RGB |
| NIR | .1304 / .9965 | **.1230 / .9970** | .1794 / .9935 | NIR |
| TIR | **.1462 / .9962** | .1550 / .9952 | .1622 / .9954 | **RGB（不是 TIR）** |

- TIR 查询上，TIR head 的 CE **.1622** 高于 RGB head 的 **.1462**；acc **.9954** 也低于 RGB head 的 **.9962**。
- 也就是说：**TIR head 不是 TIR 查询的最佳分类器。** 这是 gallery-conditioned 设计最直接的失败证据。

### 4.2 home vs cross

| q | home CE/acc | cross CE/acc | 结论 |
|---|---:|---:|---|
| RGB | .0901/.9982 | .1366/.9960 | home 更好，但 cross 只低 0.23pp acc |
| NIR | .1230/.9970 | .1549/.9950 | home 更好，但 cross 只低 0.20pp acc |
| TIR | .1622/.9954 | **.1506/.9957** | **home 反而更差** |

- 三个 head 在 train 上都接近 99.5% acc；cross head 与 home head 的差距只有 0.2–0.7pp。
- 这说明 heads 学到的是**通用身份分类器**，不是“对某个 gallery 模态特别有用”的判别边界。

### 4.3 head × P_scene 完整 cos 矩阵

| head | diag `cos(W_m,P_scene[:,m])` | best off-diagonal | 差值 | best scene |
|---|---:|---:|---:|---|
| RGB head | .4678 | .4528 (NIR) | +.0150 | RGB |
| NIR head | .4620 | .4413 (RGB) | +.0207 | NIR |
| TIR head | .4107 | **.4163 (NIR)** | **−.0056** | **NIR** |

- RGB/NIR head 的 diagonal 只比 best off-diagonal 高 **.015/.021**，非常弱。
- TIR head 的 best scene 是 **NIR**，diagonal 反而低 .0056。
- `head_id_cos`：.4748/.4677/.4369；**三个 head 都更接近 `P_id` 而不是自己的 `P_scene` 列**（尤其 TIR head：.4369 vs .4107）。
- `head_pair_cos`：RGB-NIR **.756**，RGB-TIR .459，NIR-TIR .439；RGB/NIR 两个 head 近似重复，TIR head 是一个偏 IR 的独立残差。
- `head_norm_mean`：.8105/.8106/.8108；没有 norm 塌缩；问题在方向，不在尺度。

**结论：** heads 没有形成三个 gallery-specific 方向；更准确的描述是“两个近重复的通用 identity classifier + 一个偏 IR 的残差”。TIR head 没有对准 TIR。

---

## 5. Gradient proxy：GAC 的有效影响很小且部分冗余

> 同一 checkpoint；1000 与 GD 的 head/bank 诊断完全相同；`gac_grad.json` 的差异来自 one-batch（32 PKM/96 图）采样，不是协议效应。

| 指标 | 1000 proxy | GD proxy |
|---|---:|---:|
| `share_aux` | 3.05% | 4.99% |
| `share_base` | 75.14% | 58.38% |
| `share_scene` | 15.97% | 26.77% |
| `share_id` | 5.76% | 9.74% |
| `share_pca` | 0.07% | 0.12% |
| `cos_aux_id` | .407 | .457 |
| `cos_aux_scene` | .384 | .455 |
| `cos_aux_base` | .144 | .330 |
| `cos_aux_pca` | .428 | .344 |
| `own_ratio` | [.969,.938,1.0] | [.938,.906,1.0] |
| per-head effective | 1.14/1.17/1.38% | 2.06/1.88/1.94% |

- `share_aux` 约 **3–5%**，与训练日志的 ~3.35% 一致；名义 cross 2/3 → **有效 cross 约 2–3.3%**；单个 head 约 **1–2%**。
- 三个 per-head effective 之和（3.69%/5.88%）**不等于** `share_aux`（3.05%/4.99%），因为 per-head 是各自 loss 的梯度，`share_aux` 是 mean loss 的梯度；不能把三者相加当分解。
- `cos_aux_id/scene` .38–.46、`cos_aux_pca` .34–.43：aux 梯度并非正交新方向，而是与 GPD/PCA 部分同向；**GAC 更像弱正则，而不是新的跨模态方向。**
- TIR `own_ratio=1.0`：GPD scene CE 对 TIR rows 永远选 TIR 自己的 sub-center；cross sub-centers 从不是 TIR 的正样本。GAC 是另一个 classifier，不会改变这一点。

---

## 6. Bank geometry：仍是 V2 的 class-mean 几何

来自 full checkpoint（epoch 120）：

| 指标 | RGB | NIR | TIR |
|---|---:|---:|---:|
| `cos(P_id,P_scene[:,m])` | .9567 | .9594 | **.9194** |
| `P_scene` pair cos | RGB-NIR .9157 | RGB-TIR .8014 | NIR-TIR .8098 |
| `cos(P_id,mean P_scene)` | **.9993** | | |
| bank vs recomputed class mean | .9896 | .9886 | .9878 |
| `own_ratio` | .969/.938 | .938/.906 | **1.0/1.0** |

- `P_id` 与 `mean(P_scene)` cos .9993：HPD 的层级几乎退化成一个共享 class mean；这是 V2 已有的几何，GAC 没有改变。
- TIR 的 `P_id-P_scene` 只有 .9194，低于 RGB/NIR 的 .957/.959；TIR 的 class-mean 本身更不“身份一致”。
- bank 与重算 class mean 的 cos .988–.990：full checkpoint bank 与 train class mean 一致，没有 bank 损坏；GAC 没有改写 bank。
- `own_ratio=1.0`（TIR）与 cross 尾部最差直接对应：TIR 的 cross 正样本从未进入 scene CE。

---

## 7. 为什么“只有 RGB-RGB/TIR-TIR”相对 paper 上涨

### 7.1 先看参照系

- vs UAD H014：V4 九对 **9/9 R1、9/9 mAP** 正。
- vs V2：V4 **7/9 R1、8/9 mAP** 正。
- vs **paper+1**：只有 **RGB→RGB 完整过**；**TIR→TIR 只过 R1（+0.03），mAP 差 0.44**；其余 7 对 R1/8 对 mAP 不足。
- 所以“只有 RGB/TIR”是 **paper+1 参照系**，不是 V4 的普遍事实。

### 7.2 TIR-TIR 并不是“强”，只是 paper 的 TIR 门槛低

| Pair | V4 R1/R5/R10 | paper+1 R1/mAP | 结论 |
|---|---:|---:|---|
| RGB→RGB | 43.14/61.12/68.43 | 41.47/27.83 | 完整过；R5/R10 也最强 |
| NIR→NIR | 29.56/47.17/55.97 | 30.79/18.93 | R1/mAP 都不过 |
| TIR→TIR | 22.01/37.61/46.79 | 21.98/15.00 | 只过 R1；R5/R10 比 NIR→NIR 更差 |

- TIR→TIR 的 R1 只比 paper+1 高 **0.03pp**，mAP 低 **0.44pp**；R5/R10 只有 37.61/46.79，明显低于 NIR→NIR 的 47.17/55.97。
- **“TIR-TIR 上涨”是对 raw paper 的低 TIR 基线而言；绝对 tail 上 TIR 仍弱于 NIR。**
- 之前分解：TIR-TIR mAP vs paper +0.56 = V2 baseline +0.55 + GAC +0.01；RGB-RGB mAP +1.05 = V2 +0.69 + GAC +0.36。**GAC 真正买到 mAP 的只有 RGB same；TIR same 的涨幅主要来自 V2。**

### 7.3 TIR cross 是真正的缺口

- TIR cross R5 **14.94–19.82**、R10 **21.34–27.13**、>10 **72.9–78.7%**；RGB↔NIR cross R5 38–45、R10 44–53。
- TIR head 的 best scene 是 NIR，TIR home CE 比 RGB head 差；`own_ratio=1.0`。
- 结论：**TIR 的问题不是“缺一个 head”，而是 TIR 表征 + tail/排序**；GAC 的 classifier-level 机制不可能把 TIR cross 的 >10 从 ~76% 拉下来。

---

## 8. GD 与 AG 的结构

### 8.1 GD

- 同一 checkpoint、同一 train split；head/bank 诊断完全相同。GD 的差异只来自 query/gallery 子集（399/6836/模态，984 valid）。
- GD mixed：R1 31.20（比 1000 +0.40），R5/R10 45.93/53.86（比 1000 −2.05/−2.07），>10 46.14%。
- GD pair：same R1 31.71/R5 45.93/R10 54.88；cross 13.87/24.24/30.44；cross >10 **69.56%**。
- GD TIR cross R5 **14.94–19.82**、R10 **21.34–27.13**；TIR same 19.51/31.40/42.38，弱于 1000。
- 相对 V2 的官方 GD：R1 −0.41、R5 **+1.11**、R10 −0.31——GD 不是全面退化，而是 **top1/tail/coverage 的 trade**；诊断显示 tail 的结构性弱点在 TIR cross 和 TIR same。

### 8.2 AG

- A→G：R1 35.26/R5 52.44/R10 60.07；G→A：31.83/49.53/57.97。
- 模态：
  - A→G RGB 44.22/63.11/69.11；NIR 38.22/55.33/63.33；TIR **23.33/38.89/47.78**。
  - G→A RGB 40.53/57.45/63.82；NIR 32.76/52.33/61.65；TIR **22.20/38.82/48.45**。
- AG 的 R1/R10 已过 paper+1、mAP 差 0.30（≈4–6 query，near noise）；新 R5 显示 **TIR 是 AG 的主要拖累**，RGB/NIR AG 已经较强。
- top1 错误中，A→G/G→A 约 **62.5–62.7%** 是**同模态**错误，只有 ~37% 是跨模态；平台残差主要是**模态内**平台差异，不是单纯跨模态混淆。平台机制若要独立做，应针对模态内平台不变性。

---

## 9. 机制设计 vs 代码实现

### 9.1 训练代码：正确

- full checkpoint epoch=120，bank 精确；head CE/acc 全量 train；rank 与官方完全一致；没有训练/推理 bug。
- `gac_grad` 是 32 PKM/96 图 proxy；gradient 函数只取 last block+norm，和训练诊断同口径。

### 9.2 机制设计：主因

1. **loss 对 head 置换对称**：`L_aux = mean_{n,m} CE(f_n,W_m[y])`；没有项把 head m 绑定到 gallery m，也没有 routing/weight。heads 只靠 P_scene 初始化不同；训练后都变成通用 identity classifier。
2. **heads 收敛/错位**：RGB/NIR head 近重复（pair cos .756）；TIR head best scene=NIR，TIR home CE 比 RGB head 差；三个 head 都不是 gallery-specific。
3. **有效梯度小且冗余**：aux 3–5%，cross 名义 2/3 → 2–3.3%；`cos_aux_id/scene` .38–.46、`cos_aux_pca` .34–.43；不是正交新方向。
4. **推理丢弃 heads**：GAC 只能通过共享 pre-BN feature 的梯度起作用；不能改 gallery 特征、不能做 pair routing/ranking；TIR cross 的 tail 不会因多一个 classifier 改变。
5. **bank 不变**：P_id=mean P_scene .9993；TIR own_ratio=1.0；GAC 没有改 class-mean bank，也没有给 TIR cross 正样本。

### 9.3 结论

- **不是代码没改彻底，是机制假设不成立**：GAC 预期“三个 head 分化成 gallery-specific 方向”，实际是“两个近重复通用头 + 一个偏 IR 残差头”；TIR head 没有对准 TIR。
- 诊断/归因现在已补齐；之前缺的 per-head CE/acc、完整矩阵、per-head grad、cos_aux_pca 都已落地。
- V4 的 R1/同模态收益是真实的；但 cross/TIR mAP 不能靠这组 heads 解决。

---

## 10. 对下一步的含义

1. **不建议按原计划直接做 λ/head-weight retry**：head3 已经证明不对准 TIR；把 λ 从 0.1 提到 0.15、home 0.5/cross 1.0，只会放大一个错位的通用分类器。除非先做 **TIR-specific head/weighting**，否则 retry 的预期仍然是 cross mAP +0.1–0.3，不是 +1.24。
2. **主缺口是 TIR cross 的 tail/表征**：TIR cross R5 15–20/R10 21–27/>10 73–79%；这需要 representation/ranking（或 TIR-specific adapter/translation），不是再加 classifier。
3. **NIR 不是 donor**：NIR 的 R5/R10 在 cross 里 38–45/44–53，明显好于 TIR；V4 是 additive，不是 V3 的 zero-sum。NIR 只是没有专门保护，不是预算被抢。
4. **GD/AG 是第二层**：GD 的 cross tail 与 TIR 同源；AG 的 TIR 也最弱。先解决 TIR，AG 可能自然受益；平台机制后置。
5. **单 seed 口径**：以上都是 seed1234 单 seed 的诊断；不宣称跨 seed 泛化；但 head3 错位、TIR cross tail 是效应量很大的结构事实，不是小数点噪声。
6. **λ=0 same-code control** 仍可做一个 1ep smoke 来关闭 code-drift 问题，但不必再花 120ep 来证明 GAC 机制；诊断已经给出否证。

---

## 11. 一句话总结

**V4 的 GAC 没有学出三个 gallery 条件方向：heads 在 train 上都是 ~99.5% 的通用身份分类器，RGB/NIR head 近重复，TIR head 的 best scene 是 NIR 且 TIR home CE 比 RGB head 更差；GAC 有效梯度仅 3–5% 且与 GPD/PCA 部分同向；bank 仍是 V2 的 `P_id=mean(P_scene)` 几何、TIR own_ratio=1.0。结果就是 RGB-RGB 和 NIR 的 top1 有小幅真实收益，TIR-TIR 只靠 V2 基线过 R1、mAP 不过，TIR cross 的 R5/R10 只有 15–20/21–27、>10 约 73–79%。下一步不应继续加 heads；应转向 TIR 表征/排序或 mixed-gallery，而不是再调 GAC λ。**
