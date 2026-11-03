# WEV4/GAC 实验结果：与 UAD、WEV1、WEV2、WEV3 全面对比

> 数据：
> - V4 1000：`CC/WEV4/logs/GAC_1000/test_log_20261102_0946.txt`、`domain_protocol_results.csv`
> - V4 GD：`CC/WEV4/logs/GAC_GD_test/test_log_20261102_0952.txt`
> - V4 训练：`CC/WEV4/logs/GAC_1000/train_log_20261101_2328.txt`
> - V1/V2/V3/UAD：各自 1000/GD/AG 日志与 `CC/WEV3/analysis_tpd_results.md`、`analysis_attribution_results.md`
> - UAD H014/H015 为内部复现；UAD paper 为论文值；paper+1 = 论文 +1.0pp
> 口径：1000/GD 为 official test；AG 为 Table 4；same/cross 为 3/6 对平均；九对为 Table 6 的 R1/mAP。
> 注意：单 seed；seed 噪声约 mAP 0.24 / R1 1.13；<0.5pp 不裁决。

---

## 0. 一句话

**V4/GAC 是目前综合最强（mAP/R1 最高，R5/R10 不是最高）的模型：1000 overall 11.56/30.80/47.98/55.93，九对平均 19.97/12.88，AG 12.20/35.26/60.07 与 14.00/31.83/57.97，GD 14.94/31.20/45.93/53.86。相对 V2：1000 mAP +0.09/R1 +0.28/R5 +0.47/R10 −0.21；九对平均 +0.50/+0.12；AG mAP +0.10/+0.07；GD mAP +0.03/R1 −0.41/R5 +1.11/R10 −0.31。相对 paper+1：1000 R1/R5/R10 已过，mAP 还差 0.44；AG R1/R10 已过，mAP 差 0.30；GD 只有 R10 过；九对只有 RGB→RGB 完整过、TIR→TIR 的 R1 过，平均还差 0.68/0.98。GAC 有 R1 信号但机制 mAP gate 未过：heads 部分独立/趋于收敛，aux 有效 share 只有约 2–6%（mean ~3.35%），late clip 100/114（V2 也是 100/114，不是 GAC 特有），cross mAP 只 +0.10；gallery-conditioned 归因未建立。**V4 是 UAD paper 水平（九对 +0.32/+0.02 vs paper；same +1.16/+0.54；cross −0.09/−0.24），不是 paper+1。**

---

## 1. 1000 AS overall

| 方案 | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|
| UAD H014 | 11.02 | 29.71 | 46.32 | 54.55 |
| WEV1 | 11.37 | 30.51 | 48.17 | 56.27 |
| WEV2 | 11.47 | 30.52 | 47.51 | 56.14 |
| WEV3 | 11.18 | 29.96 | 47.17 | 55.63 |
| **WEV4/GAC** | **11.56** | **30.80** | **47.98** | 55.93 |
| UAD paper | 11.0 | 29.5 | 46.3 | 54.8 |
| paper+1 | 12.0 | 30.5 | 47.3 | 55.8 |

V4 差值：

| 对照 | ΔmAP | ΔR1 | ΔR5 | ΔR10 |
|---|---:|---:|---:|---:|
| vs UAD H014 | +0.54 | +1.09 | +1.66 | +1.38 |
| vs WEV1 | +0.19 | +0.29 | −0.19 | −0.34 |
| vs WEV2 | +0.09 | +0.28 | +0.47 | −0.21 |
| vs WEV3 | +0.38 | +0.84 | +0.81 | +0.30 |
| vs UAD paper | +0.56 | +1.30 | +1.68 | +1.13 |
| vs paper+1 | **−0.44** | +0.30 | +0.68 | +0.13 |

**解读：**

- V4 的 **mAP/R1 是四方案最高**；R5 47.98 低于 V1 48.17（高 0.19），R10 55.93 低于 V1/V2，但仍过 paper+1。
- **对 paper+1：R1/R5/R10 已过，mAP 只差 0.44**。这是目前最接近 1000 paper+1 的模型。
- 对 V2：mAP +0.09、R1 +0.28、R5 +0.47 是正；R10 −0.21 是唯一回退。**R10 回退/增益都在 seed 噪声内**，需要 2 seeds。
- 对 V1：mAP/R1 正，R5/R10 负；V4 没有完整保住 V1 的 R5/R10 优势。
- 对 V3：四项全正，V4 明显好于 TPD。

---

## 2. same / cross

| 方案 | same R1/mAP | cross R1/mAP |
|---|---:|---:|
| UAD | 30.40/18.96 | 13.48/8.95 |
| WEV1 | 31.16/19.97 | 13.33/9.00 |
| WEV2 | 30.97/19.96 | 13.72/9.16 |
| WEV3 | 30.43/19.38 | 13.70/8.99 |
| **WEV4** | **31.57/20.13** | **14.17/9.26** |
| paper+1 | 31.41/20.59 | 15.26/10.50 |

V4 差值：

| 对照 | Δsame R1/mAP | Δcross R1/mAP |
|---|---:|---:|
| vs UAD | +1.17/+1.17 | +0.69/+0.31 |
| vs V1 | +0.41/+0.16 | +0.84/+0.26 |
| vs V2 | **+0.60/+0.17** | **+0.45/+0.10** |
| vs V3 | +1.14/+0.75 | +0.47/+0.27 |
| vs paper+1 | +0.16/−0.46 | **−1.09/−1.24** |

**解读：**

- V4 的 **same 和 cross 都是四方案最高**；same R1 甚至超过 paper+1，same mAP 只差 0.46。
- cross R1/mAP 比 V2 +0.45/+0.10，是方向性的正增益；但 cross mAP 只 +0.10，远小于 R1 +0.45。
- **cross 仍差 paper+1 1.09/1.24**；这是九对的主缺口。
- same 的增益（+0.60 R1）甚至大于 cross（+0.45 R1）；GAC 不是只对 cross 有效，也改善了 same。

---

## 3. 九对 Table 6

| Query→Gallery | V1 R1/mAP | V2 R1/mAP | V3 R1/mAP | **V4 R1/mAP** | paper+1 | V4−V2 | V4−paper+1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RGB→RGB | 42.01/27.52 | 42.20/27.52 | 41.36/26.63 | **43.14/27.88** | 41.47/27.83 | +0.94/+0.36 | **+1.67/+0.05** |
| RGB→NIR | 26.51/16.26 | 27.12/16.50 | 26.65/16.09 | **27.92/16.65** | 27.56/17.52 | +0.80/+0.15 | +0.36/−0.87 |
| RGB→TIR | 7.49/5.70 | 7.26/5.71 | 7.54/5.62 | **7.73/5.71** | 9.10/7.00 | +0.47/0.00 | −1.37/−1.29 |
| NIR→RGB | 23.79/15.81 | 25.25/16.11 | 24.31/15.76 | **25.71/16.37** | 26.53/17.58 | +0.46/+0.26 | −0.82/−1.21 |
| NIR→NIR | 29.13/17.68 | 29.60/17.80 | 28.67/17.19 | **29.56/17.94** | 30.79/18.93 | −0.04/+0.14 | −1.23/−0.99 |
| NIR→TIR | 6.89/5.21 | 7.03/5.29 | 7.03/5.28 | **6.93/5.37** | 8.78/6.87 | −0.10/+0.08 | −1.85/−1.50 |
| TIR→RGB | 7.73/5.77 | 7.59/5.98 | 8.20/5.87 | **8.29/6.03** | 10.04/7.32 | +0.70/+0.05 | −1.75/−1.29 |
| TIR→NIR | 7.59/5.25 | 8.06/5.39 | 8.48/5.34 | **8.43/5.43** | 9.57/6.69 | +0.37/+0.04 | −1.14/−1.26 |
| TIR→TIR | 22.34/14.70 | 21.12/14.55 | 21.26/14.32 | **22.01/14.56** | 21.98/15.00 | +0.89/+0.01 | +0.03/−0.44 |
| **平均** | **19.28/12.66** | **19.47/12.76** | **19.28/12.46** | **19.97/12.88** | **20.65/13.86** | **+0.50/+0.12** | **−0.68/−0.98** |

**解读：**

- V4 九对平均 **19.97/12.88**，四方案最高；比 V2 +0.50/+0.12，比 V3 +0.69/+0.42，比 V1 +0.69/+0.22；**vs UAD paper 19.65/12.86 = +0.32/+0.02（paper 水平，不是 +1）**。
- **7/9 R1、8/9 mAP 比 V2 上涨**；只有 NIR→NIR R1 −0.04、NIR→TIR R1 −0.10 小幅回退；RGB→TIR mAP 持平。
- **对 paper+1：只有 RGB→RGB 完整过；TIR→TIR 的 R1 过、mAP 差 0.44；其余 7 对 R1、8 对 mAP 不足。** 九对平均仍差 0.68/0.98。
- **TIR cross 的 R1 明显改善**：TIR→RGB +0.70、TIR→NIR +0.37、RGB→TIR +0.47；但 mAP 只 +0.05/+0.04/0.00，仍是“top-1 改善、平均排序不改善”。
- **NIR→TIR 是唯一 R1 回退的 cross 对**（−0.10）；NIR→NIR R1 也没恢复（−0.04），NIR same 仍是弱点。
- **对 UAD paper**：V4 R1 4/9 上涨（RGB→RGB、RGB→NIR、NIR→RGB、TIR→TIR），mAP 4/9 上涨；TIR cross / RGB→TIR 仍低于 paper；cross 平均 14.17/9.26 vs paper 14.26/9.50，**略低于 paper**。
- **对 UAD H014**：V4 **9/9 R1、9/9 mAP 上涨**（H014 九对见 `analysis_tpd_results.md`）。

### 3.1 为什么九对平均涨了，但 paper+1 还远

- V4 的增益主要来自 **same（RGB→RGB +0.94、TIR→TIR +0.89 R1）** 和 **部分 cross（RGB→NIR +0.80、TIR→RGB +0.70、NIR→RGB +0.46）**。
- **分解（关键）**：TIR→TIR mAP vs paper +0.56 = **V2 baseline +0.55 + GAC +0.01**；RGB→RGB mAP vs paper +1.05 = **V2 +0.69 + GAC +0.36**；TIR cross mAP 的 vs-paper deficit 主要是 V2 baseline 的 deficit，GAC 只 +0.00~+0.08。same 平均 +0.60/+0.17 = RGB same +0.94/+0.36 + TIR same R1 +0.89/+0.01 + NIR same flat；cross R1 5/6 positive、mAP tiny；NIR cross mAP 平均 +0.16，TIR cross 平均 +0.04。
- paper+1 的 cross 平均 15.26/10.50；V4 14.17/9.26，差 1.09/1.24。paper 的 TIR cross 本身很强（TIR→RGB 10.04、TIR→NIR 9.57、RGB→TIR 9.10、NIR→TIR 8.78），V4 的 TIR cross 只有 8.29/8.43/7.73/6.93。
- **GAC 没有把 TIR cross 的 mAP 推上去**：TIR→RGB 6.03（paper+1 7.32）、TIR→NIR 5.43（6.69）、RGB→TIR 5.71（7.00）、NIR→TIR 5.37（6.87）。TIR 的表征瓶颈仍在。
- **TIR own_ratio=1.0 的结构性原因**：HPD scene CE 对 TIR rows 实际选中的是 TIR 自己的 sub-center，cross sub-centers 从来不是 TIR 的正样本；GAC 加的是另一组 classifier，不是 cross-modal positive/ranking；所以 TIR cross R1 可以靠 top-1 边界改善，mAP 不行。
- 不能只用 overall/AG 的增益掩盖九对 cross 的不足；九对仍是主缺口。

---

## 4. AG-ReID

| 方案 | A→G mAP/R1/R10 | G→A mAP/R1/R10 |
|---|---:|---:|
| UAD | 11.46/31.63/58.52 | 13.30/30.12/54.50 |
| WEV1 | 12.02/35.11/60.22 | 13.93/32.66/57.45 |
| WEV2 | 12.10/34.74/60.89 | 13.93/32.14/57.92 |
| WEV3 | 11.61/33.33/59.93 | 13.45/31.37/57.14 |
| **WEV4** | **12.20/35.26/60.07** | **14.00/31.83/57.97** |
| paper+1 | 12.5/32.9/59.3 | 14.3/31.5/55.7 |

V4 差值：

| 对照 | A→G ΔmAP/R1/R10 | G→A ΔmAP/R1/R10 |
|---|---:|---:|
| vs UAD | +0.74/+3.63/+1.55 | +0.70/+1.71/+3.47 |
| vs V1 | +0.18/+0.15/−0.15 | +0.07/−0.83/+0.52 |
| vs V2 | **+0.10/+0.52/−0.82** | **+0.07/−0.31/+0.05** |
| vs V3 | +0.59/+1.93/+0.14 | +0.55/+0.46/+0.83 |
| vs paper+1 | **−0.30/+2.36/+0.77** | **−0.30/+0.33/+2.27** |

**解读：**

- V4 AG **mAP 最高**（12.20/14.00），比 V2 +0.10/+0.07；A→G R1 +0.52；G→A R1 −0.31；A→G R10 −0.82。
- **对 paper+1：AG 只有 mAP 差 0.30/0.30，R1/R10 都过。** AG 是三者中最接近 paper+1 的协议。
- 模态改善确实外溢到 AG；但 A→G R10 的 −0.82 是尾部代价，G→A R1 −0.31 是边界代价。
- AG 没有平台项，仍有 0.30 mAP 缺口；A→G 1350/G→A 1932 valid query，0.30%≈4–6 query，**near noise**，不是精确平台残差；V4 mAP 比 V2 只 +0.10/+0.07（≈1–1.4 query）。但比 V3 的 AG 全面回退好很多。
- A→G R10 −0.82 是尾部代价（≈11 query）；G→A R1 −0.31（≈6 query）；平台残差在 tail，不在 mAP。

---

## 5. 1000-GD

| 方案 | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|
| UAD H015 | 14.40 | 31.61 | 46.14 | 53.56 |
| WEV1 | 14.84 | 31.10 | 44.61 | 53.76 |
| WEV2 | 14.91 | 31.61 | 44.82 | 54.17 |
| WEV3 | 14.70 | 30.08 | 46.44 | 55.08 |
| **WEV4** | **14.94** | 31.20 | 45.93 | 53.86 |
| paper+1 | 15.5 | 32.9 | 46.0 | 53.5 |

V4 差值：

| 对照 | ΔmAP | ΔR1 | ΔR5 | ΔR10 |
|---|---:|---:|---:|---:|
| vs UAD | +0.54 | −0.41 | −0.21 | +0.30 |
| vs V1 | +0.10 | +0.10 | +1.32 | +0.10 |
| vs V2 | +0.03 | −0.41 | +1.11 | −0.31 |
| vs V3 | +0.24 | +1.12 | −0.51 | −1.22 |
| vs paper+1 | **−0.56** | **−1.70** | −0.07 | +0.36 |

**解读：**

- V4 GD **mAP 最高（14.94）**；R1 31.20 低于 UAD/V2 31.61，高于 V1/V3；R5 45.93 低于 V3/UAD，高于 V1/V2；R10 53.86 低于 V2/V3。
- **GD 是“mAP 最好、R1/R5/R10 混合”**：mAP 从 V2 +0.03，R1 −0.41，R5 +1.11，R10 −0.31；相对 V3 mAP/R1 正，R5/R10 负。
- 对 paper+1：**只有 R10 过**；mAP/R1/R5 都差（0.56/1.70/0.07）。GD 仍是最大缺口。
- V4 没有保住 V3 的 GD R5/R10 优势；V3 的 R5 46.44/R10 55.08 换成 V4 的 mAP 14.94/R1 31.20。是 trade，不是全面胜。

### 5.1 GD 九对

| Query→Gallery | V1 R1/mAP | V2 R1/mAP | V3 R1/mAP | **V4 R1/mAP** | V4−V2 |
|---|---:|---:|---:|---:|---:|
| RGB→RGB | 44.21/38.52 | 44.51/38.54 | 43.60/38.14 | **44.82/38.88** | +0.31/+0.34 |
| RGB→NIR | 26.52/20.72 | 26.22/20.79 | 26.22/20.62 | **28.66/20.91** | +2.44/+0.12 |
| RGB→TIR | 7.93/8.23 | 6.71/8.23 | 7.32/8.01 | **8.84/8.21** | +2.13/−0.02 |
| NIR→RGB | 22.87/19.42 | 24.39/19.69 | 23.17/19.40 | **24.39/20.10** | 0.00/+0.41 |
| NIR→NIR | 31.10/24.87 | 30.79/24.69 | 29.88/23.97 | **30.79/24.49** | 0.00/−0.20 |
| NIR→TIR | 7.93/7.55 | 8.23/7.67 | 8.23/7.69 | **6.10/7.41** | −2.13/−0.26 |
| TIR→RGB | 8.23/8.53 | 7.01/8.63 | 8.23/8.47 | **7.62/8.94** | +0.61/+0.31 |
| TIR→NIR | 7.62/7.42 | 7.62/7.27 | 8.23/7.42 | **7.62/7.27** | 0.00/0.00 |
| TIR→TIR | 20.43/15.37 | 20.73/15.40 | 20.43/15.29 | **19.51/15.07** | −1.22/−0.33 |
| **平均** | **19.65/16.74** | **19.58/16.77** | **19.48/16.56** | **19.82/16.81** | **+0.24/+0.04** |

- GD **pair 平均 V4 最高（19.82/16.81）**；但 overall R1 31.20 低于 V2 31.61，说明 **mixed-gallery** 仍拖 R1。
- GD 的 **NIR→TIR R1 −2.13、TIR→TIR −1.22** 是明显回退；GD 的 RGB/NIR 正向改善（RGB→NIR +2.44、RGB→TIR +2.13）抵消。
- GD R5/R10 低于 V3，是 pair/gallery 覆盖 trade，不是代码 bug。

---

## 6. 详细 domain（V4 vs V3）

| 类别/对 | V3 R1/mAP | V4 R1/mAP | Δ |
|---|---:|---:|---:|
| same RGB-G→RGB-G | 40.86/28.62 | 42.78/29.55 | +1.92/+0.93 |
| same NIR-G→NIR-G | 26.16/17.41 | 26.82/17.86 | +0.66/+0.45 |
| same TIR-G→TIR-G | 22.45/15.65 | 22.91/15.89 | +0.46/+0.24 |
| platform RGB-G→RGB-A | 41.93/31.56 | 41.46/32.64 | −0.47/+1.08 |
| platform RGB-A→RGB-G | 44.67/27.70 | 45.56/29.65 | +0.89/+1.95 |
| platform NIR-G→NIR-A | 32.76/24.26 | 35.56/25.88 | +2.80/+1.62 |
| platform NIR-A→NIR-G | 35.11/19.87 | 38.00/21.07 | +2.89/+1.20 |
| platform TIR-G→TIR-A | 22.36/16.22 | 22.52/16.41 | +0.16/+0.19 |
| platform TIR-A→TIR-G | 22.00/14.36 | 24.44/14.46 | +2.44/+0.10 |
| modality ground RGB→NIR | 26.75/17.52 | 28.15/17.85 | +1.40/+0.33 |
| modality ground NIR→RGB | 24.90/17.11 | 26.16/17.66 | +1.26/+0.55 |
| modality ground RGB→TIR | 7.28/6.21 | 7.42/6.31 | +0.14/+0.10 |
| modality ground TIR→RGB | 8.21/6.68 | 9.01/6.88 | +0.80/+0.20 |
| modality ground NIR→TIR | 6.56/5.67 | 6.29/5.65 | −0.27/−0.02 |
| modality ground TIR→NIR | 9.21/6.03 | 9.14/6.07 | −0.07/+0.04 |

- V4 的 same/platform/大部分 modality 都优于 V3；**NIR→TIR 仍小退**。
- detailed domain 与九对/AG 的结论一致：same/platform 强，NIR/TIR 的个别方向弱。

---

## 7. 训练日志与 GAC 机制

### 7.1 基础事实

- `GAC enabled: weight target 0.100, ramp 5, M=3`；`GAC heads initialized from P_scene: M=3, P_id fallback rows=0`。
- `aux_w`：epoch1 0.050 → epoch5 0.100，之后 constant。
- 训练 120 epoch；无 NaN；bank counts 三模态 500；HPD warm start 500/500；inference 不变。

### 7.2 GAC diagnostics

| 阶段 | loss_aux | share_aux | head_scene_cos | head_pair_cos | cos aux-id/scene/base |
|---|---:|---:|---|---|---|
| epoch1 | 5.432 | .023 | [.733,.735,.750] | [.689,.308,.290] | .258/.218/.042 |
| epoch~10 | 2.438 | .020 | [.361,.312,.190] | [.699,.329,.308] | .680/.542/.247 |
| epoch109 | .100 | .029 | [.468,.461,.410] | [.756,.459,.439] | .586/.507/.194 |
| epoch120 | .109 | .043 | [.468,.462,.411] | [.756,.459,.439] | .606/.492/.312 |

**解读：**

- `head_scene_cos` 从 ~0.73 降到 ~0.46，heads **明显偏离 P_scene**；不是 V3 的固定 class mean；`head_pair_cos` [.756,.459,.439] 说明 heads **没有塌缩**，学出了不同方向。
- `loss_aux` 5.43→0.11；heads 确实在学习。
- `cos_aux_id/aux_scene` late ~0.5–0.6，`cos_aux_base` ~0.2–0.3；aux 梯度与 id/scene 仍部分同向，但 base 上有独立分量。
- **`share_aux` 只有约 2–6%**（目标 5–10%）；late `clip_count` 100/114，但 **V2 末期也是 100/114**（clip_count 是 epoch 内累计；V2 iter50=50、iter100=100）；pre-clip mean V2 2.20 vs V4 2.35；**clip 不是 GAC 特有**，不能拿 V2 “50/114” 解释 GAC 的弱效果。aux 的有效特征影响仍小。这解释了：
  - 1000 cross R1 +0.45 但 mAP 只 +0.10；
  - same +0.60 R1/+0.17 mAP；
  - top-1 改善 > 平均排序；
  - R10 小幅回退。
- **rank bins（由 aggregate 反推，±1 query）**：1000 V4 **1973/1100/509/2823** vs V2 **1955/1088/553/2809**；V4−V2 **rank1 +18、2–5 +12、6–10 −44、>10 +14**。mAP +0.09 是 top 增益与 **6–10 −44** 的净结果；不是纯 top-1 故事。GD V4 **307/145/78/454** vs V2 **311/130/92/451**；rank1 −4、2–5 +15、6–10 −14、>10 +3；GD 是 coverage/boundary trade。
- **负 logits 锐化**：V4 late id pos/neg/entropy 25.234/**0.129**/0.074、scene 26.203/**1.107**/0.061 vs V2 25.250/**0.858**/0.082、26.219/**1.785**/0.066；正 logits 基本不变（~25.2/26.2）。这是 margin sharpening，不是特征对齐；它更容易改 top-1 而不是 AP。
- **HPD geometry 几乎没变**：late `cos(P_id,P_scene)` [.957,.959,.919]、pair cos [.916,.801,.810]、`cos(P_id,mean P_scene)=.999`；V2 是 [.956,.959,.919]、[.914,.800,.809]。**GAC 没有改变 P_id/P_scene 的 class-mean 几何**；它是在这个 bank 之外加了一个 learned classifier direction。
- `own_ratio` late [.906,.938,1.0]；TIR 1.0；TIR 保护仍在；bank usage [62,66,64] 平衡。
- final train loss：base .087、WRT .111、PCA .205、GPD .066、aux .331（raw；加权 .033）；V2 base .084、WRT .115、PCA .205、GPD .069；训练收敛正常。
- **诊断缺口（P1，归因层）**：`head_scene_cos` 只有对角 `cos(W_m,P_scene[:,m])`；没有完整 `cos(W_m,P_scene[:,m'])` 矩阵、per-head CE/acc、home/cross accuracy、per-head gradient share、`cos(g_aux,g_pca)`、feature norm；没有 matched shared W_aux control / head ablation。因此“gallery conditioning”的因果不可严格识别；“heads 学到方向”只是几何证据。
- **结论**：GAC 是“真实但弱、部分冗余”的辅助分类器，不是已证实的 gallery 条件跨模态方向；heads 没有塌缩，但 head_pair 从 [.689,.308,.290] 升到 [.756,.459,.439]（趋于收敛；两对 <0.5），aux 梯度与 GPD id/scene 的 cos 从 .258/.218 升到 .606/.492；有效 share 2–6%（mean ~3.35%）；cross mAP gate 失败。下一步若继续 GAC，应先补 per-head CE/acc/grad + shared control，再决定是否 λ/头权重微调；但 classifier-level 的上限仍在。
- **λ=0 anchor**：代码路径正确（不 forward/loss、heads 不初始化、strict V2），但交付日志没有 λ=0 smoke；第一次 run 在 epoch1 崩溃前无 checkpoint，**“strict V2”是代码验证，不是本次实证**。
- **2337 P1**：`PKMSampler_DDP.sample_list` 对缺模态 pid 会 `divmod(need, 0)`（`datasets/sampler_ddp.py`）；README 已警告，但代码无 fallback；2337 必须先做缺模态预检，不能直接跑。

---

## 8. Gate / paper+1 判定

### 8.1 V4 设计 gate（不是 V3 W/G）

V4 方案预注册的是 **机制信号 gate**：

| 条件 | 要求 | 实际 | 结果 |
|---|---:|---:|---|
| cross 6 对 R1 | ≥ V2+0.4（14.12） | 14.17 | **pass** |
| cross 6 对 mAP | ≥ V2+0.3（9.46） | 9.26 | **fail** |
| 九对平均 R1 | ≥ V2+0.2（19.67） | 19.97 | **pass** |
| 九对平均 mAP | ≥ V2+0.2（12.96） | 12.88 | **fail** |
| 单对 | 无 pair 低于 V2−0.5 | 最低 NIR→NIR R1 −0.04、NIR→TIR −0.10 | pass |
| same | ≥ V2−0.3（30.67/19.66） | 31.57/20.13 | pass |
| AG mAP | ≥ V2−0.3（11.80/13.63） | 12.20/14.00 | pass |
| GD mAP | ≥ V2−0.3（14.61） | 14.94 | pass |
| overall mAP | ≥ V2（11.47） | 11.56 | pass |
| R5/R10 | ≥ V2−0.5（47.01/55.64） | 47.98/55.93 | pass |
| 2 seeds | 必须有 | 只有 seed1234 | **未满足** |

**结论：V4 预注册机制 gate 在 mAP 上未通过；R1 过，但 R1 增益 +0.45 仍低于 <0.5pp 噪声线；2 seeds 缺失。** 因此不能写“GAC direction holds”；正确说法是“GAC 有 R1 信号，但 mAP 机制信号失败，需 2 seeds/更强机制”。

**参考（不是 V4 gate）**：V4 相对 V3 的 W/G 会 pass G screening、W 只差 overall R10 −0.21 与 NIR→TIR R1 −0.10；但这是 V3 的 gate，不等于 V4 的预注册 gate。

### 8.2 paper+1

| 协议 | V4 | paper+1 | 结果 |
|---|---:|---:|---|
| 1000 | 11.56/30.80/47.98/55.93 | 12.0/30.5/47.3/55.8 | **R1/R5/R10 过，mAP 差 0.44** |
| GD | 14.94/31.20/45.93/53.86 | 15.5/32.9/46.0/53.5 | 只有 R10 过；mAP/R1/R5 差 0.56/1.70/0.07 |
| AG A→G | 12.20/35.26/60.07 | 12.5/32.9/59.3 | R1/R10 过，mAP 差 0.30 |
| AG G→A | 14.00/31.83/57.97 | 14.3/31.5/55.7 | R1/R10 过，mAP 差 0.30 |
| 九对平均 | 19.97/12.88 | 20.65/13.86 | 差 0.68/0.98 |
| 九对 cross | 14.17/9.26 | 15.26/10.50 | 差 1.09/1.24 |

- **V4 在 1000 上过 paper+1 三项（R1/R5/R10）**；V1/V2 也过这三项，V4 的 mAP 更高（11.56 vs 11.47/11.37），mAP 只差 0.44。
- **AG 只差 mAP 0.30**；R1/R10 都过。
- **GD 仍只有 R10 过**；R1 差 1.70，是最大硬缺口。
- **九对只有 RGB→RGB 完整过；TIR→TIR R1 过、mAP 差 0.44；其余 7 对不足。** 九对/VI 仍是主缺口。

---

## 9. 统计边界

- 单 seed；seed 噪声约 mAP 0.24/R1 1.13。
- V4 vs V2：1000 mAP +0.09/R1 +0.28/R5 +0.47/R10 −0.21，**在噪声内**；不能宣布 V4 一定赢 V2，但方向一致（mAP/R1/R5 正）。
- V4 vs V1：mAP +0.19/R1 +0.29，**在噪声内**；R5/R10 负 0.19/0.34，也在噪声内。
- V4 vs V3：mAP +0.38/R1 +0.84/R5 +0.81/R10 +0.30，R1 仍 <1.13；但四项同向，比 V3 更可信。
- 九对平均 +0.50 R1/+0.12 mAP；cross +0.45/+0.10；same +0.60/+0.17；单 seed 不能裁决，需 2 seeds。
- GD **984 valid query**（399/模态是 same-camera 排除前；984 是 GD overall 的有效 query）：1pp≈9.84 query；V4 GD R1 −0.41 vs V2 ≈4 query，GD R5 +1.11 ≈11 query，GD R10 −0.31 ≈3 query；都不稳。此前“399 query/1.5pp≈6 query/2–3 query”的口径偏小，已纠正。
- AG 1350/1932 query；A→G R1 +0.52、R10 −0.82；需 seeds。
- **本轮执行口径（2026-11-02 用户决定）：指标强弱分析只做 seed1234 单 seed**；V1–V4 既往实验均为同一 seed，方法间按 same-seed paired 对比；2 seeds 后置，不作为本轮 gate 前置。单 seed 结论只对 seed1234 成立；用效应量、12-checkpoint 一致性、query-level bootstrap 代替跨 seed 显著性。
- **epoch 稳健性（12 次 eval，epoch10–120）**：V4−V2 mAP 12/12 正、R1 11/12、R5 7/12、R10 6/12；V4−V1 mAP 11/12、R1 8/12、R5 **1/12**、R10 2/12；V4−V3 **48/48**（4 指标×12 eval）。解释：V4 vs V2 是小而一致的 mAP/R1 offset，不是最后一 epoch 偶然；但 R5/R10 后期转负（R10 epoch80 +0.47 → epoch90 +0.06 → epoch100 −0.26 → epoch120 −0.21）；V4 vs V1 是“V4 mAP/R1、V1 R5/R10”的 trade；V4 vs V3 清楚但约 2/3 是 V2 复位。12 个 checkpoint 相关，不是 12 个独立 seed。

---

## 10. 结论：V4 的定位

**正面：**

- 1000 overall **mAP/R1 最高**；R5/R10 不是最高（V1 更高），但 R1/R5/R10 都过 paper+1；mAP 只差 0.44。
- 九对平均 **19.97/12.88 最高**；cross 平均 **14.17/9.26 最高**；same 平均 **31.57/20.13 最高**。
- AG **mAP 最高**（12.20/14.00），只差 paper+1 0.30；R1/R10 已过。
- GD **mAP 最高（14.94）**；GD pair 平均最高（19.82/16.81）。
- 对 V3：overall/same/cross/AG 全面正；对 UAD H014：9/9 pair R1/mAP 正。
- 机制上：GAC heads 偏离 P_scene（~0.46），head_pair [.756,.459,.439]；不是 V3 同 bank，但 head_pair 从 [.689,.308,.290] 升到 [.756,.459,.439]，两对 <0.5，heads 趋于收敛；**gallery-conditioned 归因未建立，机制 mAP gate 失败**。

**不足：**

- 1000 mAP 还差 paper+1 0.44；GD R1 差 1.70、R5 差 0.07、mAP 差 0.56；AG mAP 差 0.30；九对平均差 0.68/0.98，cross 差 1.09/1.24。
- 九对仍只有 RGB→RGB 完整过；TIR cross/RGB→TIR/NIR→TIR mAP 仍远低于 paper；NIR→NIR/NIR→TIR 是弱点。
- GAC 的有效 share 只有 2–6%；late clip 100/114，但 V2 也是 100/114，pre-clip V2 2.20 vs V4 2.35，clip 不是 GAC 特有；cross mAP 只 +0.10；top-1 改善 > 平均排序。
- 相对 V1：R5/R10、AG G→A R1、TIR→TIR 有回退；相对 V2：R10、GD R1/R10、AG A→G R10 有回退。V4 是 trade，不是全面支配。
- 单 seed；V4 vs V2/V1 的 1000 差异在噪声内；需 2 seeds。
- 2337 未跑。

**下一步（只给方向）：**

1. **本轮不跑 seed 5678**；按用户口径用 seed1234 单 seed 做 same-seed paired 分析；用效应量、12-checkpoint 一致性、query-level bootstrap 判强弱；多 seed 后置。
2. GAC 若继续：先补 **per-head CE/acc/grad + shared W_aux control**，再考虑 λ/头权重；当前 heads 部分独立/趋于收敛，share 小，且 mAP gate 失败；不要用“heads 已独立”作为继续加模块的理由。
3. 九对 cross：TIR cross mAP / NIR→NIR / NIR→TIR 仍是主缺口；需要 representation/ranking，而不是再调同一 class-mean bank。
4. GD：mAP 已最好，但 R1/R5/R10 混合；mixed-gallery 仍是 GD R1 的主因。
5. AG：只差 mAP 0.30；平台机制仍可后置。
6. 2337：等 1000 2 seeds/W gate 决策后再跑。

---

## 11. 一句话

**V4/GAC 是目前综合最强（mAP/R1 最高）的模型，但不是四项全最高：1000 R1/R5/R10 过 paper+1、mAP 只差 0.44；AG 只差 mAP 0.30；GD mAP 最高但 R1/R5/R10 混合；九对平均/cross/same 都是最高，但九对仍只有 RGB→RGB 完整过、cross 差 paper+1 1.09/1.24。GAC heads 部分独立、偏离 P_scene，但 head_pair 趋于收敛（两对 <0.5），有效 share 只有 2–6%，cross mAP 只 +0.10，机制 mAP gate 失败；gallery-conditioned 归因未建立（无 per-head CE/grad、无 shared control）。下一步用 seed1234 单 seed + 诊断/λ=0 control 裁决，再决定是否加强 GAC 的有效 cross 梯度或转向 TIR/ranking；多 seed 后置。**
