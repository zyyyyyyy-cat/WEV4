# WEV4/GAC 方案设计与代码深度剖析

> 数据：`CC/WEV4/analysis_gac_results.md`、`CC/WEV4/logs/GAC_1000/train_log_20261101_2328.txt`、`CC/WEV4/logs/GAC_1000/test_log_20261102_0946.txt`、`CC/WEV4/logs/GAC_GD_test/test_log_20261102_0952.txt`
> 代码：`CC/WEV4/WHU`（基于 `CC/WEV2/WHU`），对照 `CC/WEV1/WHU`、`CC/WEV3/WHU`、`CC/WHU`
> 标记：[代码事实] / [实验支持] / [机制假设] / [统计边界]。单 seed；<0.5pp 不裁决。

---

## 0. 结论摘要

**V4/GAC 的设计目标是“V2 复位 + gallery 条件辅助分类器”，用三个从 P_scene 初始化、作用在所有 query 模态上的 learned heads，补 cross 的判别边界；实际指标是“1000 mAP/R1 最高、same/cross 最高、AG mAP 最高、GD mAP 最高”，但九对 cross mAP 只 +0.10、九对 paper+1 仍差 0.68/0.98、GD R1 差 1.70、AG mAP 差 0.30。**

核心判断：

- **V4 是 UAD paper 水平，不是 paper+1**：九对 19.97/12.88 vs UAD paper 19.65/12.86 = +0.32/+0.02；same +1.16/+0.54；cross −0.09/−0.24；vs paper+1 九对 −0.68/−0.98、cross −1.09/−1.24。
- **训练代码是正确的**：aux_heads 注册/初始化/loss/target/DDP/AMP/λ=0/推理/resume 都按设计实现；没有训练 bug。
- **V4 预注册机制 gate 在 mAP 上失败**：cross R1 +0.45 过 +0.4，但 mAP +0.10 < +0.30；九对 R1 +0.50 过 +0.2，但 mAP +0.12 < +0.20。2 seeds 是原 gate 要求，但 **2026-11-02 用户决定本轮只做 seed1234 单 seed，多 seed 后置**；R1 信号仍低于 <0.5pp 噪声线；**不能写“GAC direction holds”**。
- **机制设计是主要瓶颈**：GAC 是“同一 pre-BN 特征上的额外 classifier family”，不是 feature alignment/translation；heads 偏离 P_scene（head_scene_cos ~0.46），但 head_pair 从 [.689,.308,.290] 升到 [.756,.459,.439]（两对 <0.5，趋于收敛），不是三个 gallery-specific 独立方向；aux 有效 share 2–6%；HPD bank geometry 与 V2 几乎不变；所以它主要改善 top-1/same，不能把 cross mAP 推上去。
- **“只有 RGB-RGB/TIR-TIR”是 vs UAD paper Table 6 的参照系现象**：vs UAD H014 是 9/9 R1、9/9 mAP 上涨；vs V2 是 7/9 R1、8/9 mAP 上涨；但 vs paper，只有 RGB-RGB +2.67/+1.05、TIR-TIR +1.03/+0.56 是 R1/mAP 双涨；RGB-NIR R1 +1.36 但 mAP +0.13；NIR-RGB R1 +0.18/mAP −0.21；TIR cross 仍低于 paper。paper 的 cross 本身很强，V4 cross 平均 14.17/9.26 vs paper 14.26/9.50，仍略低于 paper。
- **代码是否有“不够彻底”**：训练路径没有；**诊断/归因不完整**——设计要求的 per-head CE/acc/grad share 没有完整落盘，也没有 matched shared W_aux control / head ablation；因此“gallery conditioning vs 额外 pre-BN classifier”的因果归因只能算部分成立，不能完全拆分。这是代码/实验完整性问题，不是训练 bug。
- **V4 vs V2/V1 是小幅/噪声内改进**：1000 +0.09/+0.28/+0.47/−0.21 vs V2；+0.19/+0.29/−0.19/−0.34 vs V1；九对 +0.50/+0.12；AG mAP +0.10/+0.07；GD +0.03/−0.41/+1.11/−0.31。V4 不是全面支配，是 trade；需 2 seeds。
- **V4 vs V3 是明确改进**：overall/same/cross/AG 全面正；GD mAP/R1 正、R5/R10 负；V4 恢复了 V2 的强项，同时 GAC 提供了部分独立但趋于收敛、弱的分类方向；gallery-conditioned 因果归因未建立。

---

## 1. V4 的方案设计目标

### 1.1 目标层级

| 层级 | 目标 |
|---|---|
| 最终目标 | 1000 AS / 1000GD / 2337 / AG / VI 九对，mAP/R1/R5/R10 全部 ≥ paper+1；mINP 次之 |
| 第一轮机制信号 | 九对 Table 6 第一，尤其 cross；AG 第二；paper+1 是 Tier2 |
| 第一轮 gate | V4 设计 gate：cross ≥V2+0.4 R1/+0.3 mAP；九对 ≥+0.2/+0.2；无 pair 低于 V2−0.5；same ≥V2−0.3；AG/GD mAP ≥V2−0.3；overall mAP ≥V2；R5/R10 ≥V2−0.5；2 seeds。V3 的 W/G 只作 reference，不是 V4 预注册 gate |
| 单机制 | 只做 GAC；不做 CMFA/per-modality/AP/adapter/platform 同轮 |
| 推理 | pre-BN CLS + L2 + 欧氏；不加载 heads/bank；单模型 |

### 1.2 设计假设

1. V2 的 `P_id`/`P_scene` 0.75/0.25 是最强基线；V3 的 gallery cross 是 same-bank 零和，不能继续。
2. 三个 **gallery 条件 learned heads** `W_m` 从 `P_scene[:,m]` 初始化，每个头作用在所有 query 模态上；global 500 类 CE；可以比 V3 的固定 class mean 更灵活。
3. 等权 `(1/M)Σ_m CE`：cross 头占 2/3、home 头占 1/3；名义上 cross-first。
4. 训练-only；heads 不参与推理；推理特征仍是 V2 的 pre-BN CLS。
5. 预期 cross +0.3~0.8 R1/+0.2~0.6 mAP；paper+1 是 Tier2，不承诺。

---

## 2. 代码实际做了什么

### 2.1 结构

- `model/make_model.py`：
  - `aux_heads = nn.Parameter(torch.zeros(M,C,768))`，`M=len(MODALITIES)=3`；在 `build_transformer.__init__` 注册（optimizer 之前）；zero/no-RNG。
  - `forward(..., return_aux=False)`：mode 0 时若 `return_aux=True`，用 **pre-BN `global_feat`**、L2、`einsum('nd,mcd->nmc')/AUX_TAU`，fp32（`autocast(enabled=False)`）；mode≥1 不变。
  - `load_param`/`load_param_finetune` 跳过 `aux_heads`；兼容 full checkpoint dict。
- `processor/processor.py`：
  - `initialize_gac_heads`：warm start + bank broadcast 后，**所有 rank** 从 `P_scene[:,m]` 填；零行 P_id fallback；全零/non-finite raise。
  - `gac_aux_weight`：0.05→0.10 ramp（epoch1→5）。
  - `gac_aux_loss`：`target_rep.unsqueeze(1).expand(-1,M).reshape(-1)` 配 `aux_scores.reshape(N*M,C)`；不是 `repeat(M)`。
  - `do_train`：λ>0 时 `model(..., return_aux=True)`；`loss += λ*L_aux`；aux meter；诊断；λ=0 严格 3 元组；推理不变；resume 完整。
  - `diagnose_gpd_gradients`：可选 `aux_weight`；计算 `g_aux/share_aux/cos_aux_*`、`head_scene_cos/head_id_cos/head_pair_cos`。
- `test.py`：未改；`load_param` 跳过 aux；mode≥1 不变。
- 代码审查/复审：P0=0/P1=0；训练路径无 bug；`GAC diag` 日志格式 bug 已修；resume 已加固。

### 2.2 实际数据/梯度流

| 阶段 | 事实 |
|---|---|
| 输入 | P identity × M 模态 × K instance；`global_feat` [M*B,768] pre-BN |
| GAC | `f=L2(global_feat)`；`W=L2(aux_heads)` [M,C,D]；`aux_scores` [M*B,M,C]；`L_aux=mean CE` 全 C 类 |
| 梯度 | `L_aux` 回传到 `aux_heads` 和 `global_feat`；heads 在 DDP forward 内；λ=0 时 heads unused |
| Bank | `P_id/P_scene` 是 buffer；GAC 不更新 bank；warm start/ProCA 更新不变 |
| 推理 | mode≥1；pre-BN CLS + L2 + 欧氏；heads 丢弃；768D |

### 2.3 训练诊断（实测）

| 阶段 | loss_aux | share_aux | head_scene_cos | head_pair_cos | cos aux-id/scene/base |
|---|---:|---:|---|---|---|
| epoch1 | 5.432 | .023 | [.733,.735,.750] | [.689,.308,.290] | .258/.218/.042 |
| epoch~10 | 2.438 | .020 | [.361,.312,.190] | [.699,.329,.308] | .680/.542/.247 |
| epoch109 | .100 | .029 | [.468,.461,.410] | [.756,.459,.439] | .586/.507/.194 |
| epoch120 | .109 | .043 | [.468,.462,.411] | [.756,.459,.439] | .606/.492/.312 |

- `head_scene_cos` 从 ~0.73 降到 ~0.46：heads **明显偏离 P_scene**；不是 V3 的固定 class mean。
- `head_pair_cos` [.756,.459,.439]：heads **没有塌缩**，学出了不同方向。
- `loss_aux` 5.43→0.11：heads 确实在学习。
- `cos_aux_id/aux_scene` late ~0.5–0.6，`cos_aux_base` ~0.2–0.3：aux 梯度与 id/scene 仍部分同向，base 上有独立分量。
- **head 几何的轨迹**：init 时 heads = P_scene，pair cos 应为 ~[.916,.801,.810]；epoch1 已降到 [.689,.308,.290]（先去相关），之后回到 [.756,.459,.439]（部分回相关）；`head_scene_cos` .733→.361→.46。heads **不是三个完全独立方向**，而是“先去相关、后部分回相关”的部分独立分类器；最终仍比 P_scene 独立，但保留 RGB/NIR 更近、TIR 更远的模态结构。**两对 late cos <0.5（.459/.439），且所有 pair 都随训练上升；heads 在收敛，不是三个 gallery-specific 方向。** 这解释了收益跟着模态结构走：RGB/NIR 更近；NIR 无专门保护/最弱（V4 additive，不是 V3 的 zero-sum donor）；TIR head 更独立、TIR-TIR 受益。
- **`share_aux` 只有约 2–6%**；late `clip_count` 100/114，但 **V2 末期也是 100/114**（clip_count 是 epoch 内累计；V2 iter50=50、iter100=100）；pre-clip mean V2 2.20 vs V4 2.35；**clip 不是 GAC 特有**，不能拿 V2 “50/114” 解释 GAC 的弱效果；aux 的有效特征影响仍小。
- **HPD geometry 几乎没变**：late `cos(P_id,P_scene)` [.957,.959,.919]、pair cos [.916,.801,.810]、`cos(P_id,mean P_scene)=.999`；V2 是 [.956,.959,.919]、[.914,.800,.809]。**GAC 没有改变 class-mean bank**。
- `own_ratio` late [.906,.938,1.0]；TIR 1.0；bank usage [62,66,64] 平衡。
- final train loss：base .087、WRT .111、PCA .205、GPD .066、aux .331（raw；加权 .033）；V2 base .084、WRT .115、PCA .205、GPD .069；收敛正常。

---

## 3. 设计目标 → 机制 → 实际

| 设计目标 | 机制 | 预设功能 | 实际 | 判定 |
|---|---|---|---|---|
| 九对 cross 第一 | gallery cross heads（2/3 aux） | 直接监督 query→gallery 模态 | cross 14.17/9.26，vs V2 +0.45/+0.10；vs paper+1 −1.09/−1.24 | **部分：R1 有，mAP 很弱** |
| same 不丢 | home head + V2 own | 保 same/R5/AG | same 31.57/20.13，vs V2 +0.60/+0.17 | **实现，且比 cross 更强** |
| overall > V2 | V2 reset + GAC | 1000 11.56/30.80 | mAP/R1 +0.09/+0.28；R5 +0.47；R10 −0.21 | **部分：3/4 正，R10 小负** |
| AG 副产物 | all-modality GAC | AG 自然提升 | A→G 12.20/35.26/60.07；G→A 14.00/31.83/57.97 | **mAP 最高；paper+1 mAP 差 0.30** |
| GD R5/R10 | GAC + V2 | 保 GD 覆盖 | GD 14.94/31.20/45.93/53.86；R5/R10 低于 V3 | **mAP 最好；R1/R5/R10 混合** |
| TIR 保护 | TIR own .80 + home | TIR same/cross | TIR→TIR 22.01/14.56（vs V2 +0.89/+0.01；vs V1 −0.33/−0.14）；TIR cross R1 涨/mAP 平 | **部分：TIR same 未回 V1，cross mAP 未解决** |
| NIR 不弱化 | 无专门项 | NIR same/cross | NIR→NIR −0.04/+0.14；NIR→TIR −0.10/+0.08；RGB→NIR +0.80，NIR→RGB +0.46 | **NIR 仍最弱、无专门保护；V4 additive，不是 V3 的 zero-sum donor** |
| R5/R10/tail | 无专门项 | 不丢覆盖 | 1000 R10 −0.21 vs V2；AG A→G R10 −0.82；GD R5/R10 低于 V3 | **top-1 重，tail 有代价** |
| 推理不变 | 无 bank/heads 推理 | 单模型 | mode≥1 与 pre-BN 一致；test 不变 | **实现** |
| 代码正确 | 7 分派/初始化/loss | 可归因 | P0/P1=0；训练路径正确 | **实现** |

---

## 4. 为什么 cross R1 +0.45，但 cross mAP 只 +0.10

### 4.1 GAC 是 classifier，不是 feature alignment

- GAC 的三个 heads 是 **同一 pre-BN 特征上的额外分类器**；训练结束后 heads 被丢弃，推理只用 `global_feat`。
- 因此 GAC 只能通过 **梯度** 改变 `global_feat`；不能直接改变 gallery 特征、不能做 pair-specific 对齐、不能做 feature translation。
- **“gallery conditioning” 只存在于训练**：heads 在训练时对 query 施加 gallery 模态的边界；推理时不加载 heads、不做 pair routing/score fusion；共享特征必须自己学出条件化；如果特征几何没变，cross 收益自然小。
- **global 500 类负样本不是新负样本**：它和 CE/GPD 一样是全部 class 的 softmax 负类；GAC 增加的是“另一组 learned classifier 参数”，不是新的 cross-modal hard negative 或 pair-level negative。V3 的 cross CE 也用了 global 500 类负样本；GAC 的不同在于 heads 可学习/可偏离，但负样本结构相同。
- 实测：HPD bank geometry 与 V2 几乎一致；`head_scene_cos` ~0.46、`head_pair_cos` .756/.459/.439，说明 heads 自己多样，但 **特征几何没有被重写**。
- cross mAP 取决于 feature 的平均排序；classifier 只能推 top-1 边界；所以 **R1 改善 > mAP**。

### 4.2 有效 share 太小 / clip 饱和

- 目标 share 5–10%；实际 `share_aux` 2–6%；late `clip_count` 100/114，但 V2 末期也是 100/114（pre-clip V2 2.20 vs V4 2.35）；**clip 不是 GAC 特有**，不能拿 V2 “50/114” 解释 GAC 的弱效果。
- aux 梯度被全局 `clip_grad_norm_(1.0)` 截断；heads 的部分独立方向没有足够有效梯度改写特征；但 clip 本身不是 GAC 独有。
- 名义 equal mean 让 cross 占 2/3 aux，但 **有效 cross share 不是 2/3**；总 aux 只有 2–6%，cross 有效份额更低。
- 这解释了 cross R1 +0.45 但 mAP +0.10；same +0.60/+0.17；top-1 改善 > 平均排序。

### 4.3 与 id/scene 仍部分同向

- `cos_aux_id/aux_scene` late ~0.5–0.6；`cos_aux_base` ~0.2–0.3。
- heads 虽然偏离 P_scene（cos .46），但其梯度仍部分与 GPD id/scene 同向；不是完全正交的 cross 方向。
- 所以 GAC 是“额外 classifier 正则”，不是“新的跨模态特征方向”；cross mAP 的增益自然有限。

### 4.4 TIR cross 的 R1/mAP trade

- TIR→RGB 8.29/6.03（vs V2 +0.70/+0.05）、TIR→NIR 8.43/5.43（+0.37/+0.04）、RGB→TIR 7.73/5.71（+0.47/0.00）、NIR→TIR 6.93/5.37（−0.10/+0.08）。
- TIR cross **R1 上去了，mAP 几乎不动**：top-1 边界改善、平均排序不改善；TIR 表征瓶颈仍在。
- TIR→TIR 22.01/14.56（vs V2 +0.89/+0.01；vs V1 −0.33/−0.14）：TIR own .80 保护了 top-1，但没有完全恢复 V1 的 TIR same；TIR representation 仍是上限。

---

## 5. 为什么“只有 RGB-RGB/TIR-TIR”上涨

### 5.1 先分清参照系

- **vs UAD paper Table 6**：V4 R1/mAP 双涨只有 **RGB-RGB +2.67/+1.05、TIR-TIR +1.03/+0.56**；RGB-NIR R1 +1.36 但 mAP +0.13；NIR-RGB R1 +0.18/mAP −0.21；TIR cross/RGB-TIR/NIR-TIR 仍低于 paper。用户观察的“只有 RGB-RGB/TIR-TIR”就是 **paper 参照系**。
- **vs UAD H014**：V4 是 **9/9 R1、9/9 mAP 上涨**。
- **vs V2**：V4 是 **7/9 R1、8/9 mAP 上涨**；只有 NIR-NIR R1 −0.04、NIR-TIR R1 −0.10 小退。
- 所以“只有两条”不是 V4 的普遍事实，而是 **vs paper** 的事实；paper 的 cross 本身很强（cross 平均 14.26/9.50），V4 cross 平均 14.17/9.26 还略低于 paper。

### 5.2 为什么是 RGB-RGB/TIR-TIR

**RGB-RGB：**

- RGB 是最强模态；V2 own 0.60 + home head + GAC 额外分类器；RGB 能吸收 cross 预算。
- V4 RGB→RGB 43.14/27.88，vs paper +2.67/+1.05；vs V2 +0.94/+0.36；四方案最高。
- 这是“强模态 + 额外 classifier capacity”的直接收益。

**TIR-TIR：**

- TIR own 0.80、own_ratio 1.0；TIR 被 V2/GAC 权重保护；home head 对 TIR 更有效。
- V4 TIR→TIR 22.01/14.56，vs raw UAD paper +1.03/+0.56；vs **paper+1** 是 **+0.03/−0.44**（R1 几乎持平、mAP 不足）；vs V2 +0.89/+0.01；但仍低于 V1 22.34/14.70（−0.33/−0.14）。
- TIR same 的 R1 收益是真实的；mAP 只 +0.01 vs V2，说明 TIR 的 AP/表征没有本质改善；**“TIR-TIR 清晰上涨”只对 raw paper 成立，对 paper+1 不成立。**

**NIR-NIR：**

- NIR own 0.60；没有专门 head/权重；NIR 无专门保护、最弱；V4 GAC 是 additive，不是 V3 的 zero-sum donor；不要直接把 V3 的 donor 结论搬过来。
- V4 NIR→NIR 29.56/17.94，vs V2 −0.04/+0.14；vs paper −0.23/+0.01；R1 没有恢复。
- NIR 的 cross 有改善（RGB→NIR +0.80、NIR→RGB +0.46），但 NIR same 没有；NIR 仍最弱、无专门保护。

**TIR cross / RGB→TIR / NIR→TIR：**

- TIR cross R1 改善，但 mAP 几乎不动；TIR 表征弱，GAC 是 classifier，不是 translation。
- **TIR own_ratio=1.0 的结构性原因**：HPD scene CE 对 TIR rows 实际选中的是 TIR 自己的 sub-center，cross sub-centers 从来不是 TIR 的正样本；GAC 加的是另一组 classifier，不是 cross-modal positive/ranking；所以 TIR cross R1 可以靠 top-1 边界改善，mAP 不行。
- paper 的 TIR cross 很强（TIR→RGB 9.04、TIR→NIR 8.57、RGB→TIR 8.10、NIR→TIR 7.78），V4 只有 8.29/8.43/7.73/6.93；仍低于 paper。
- **head geometry 也支持这个模式**：`head_pair_cos` RGB-NIR .756 > RGB-TIR .459 / NIR-TIR .439；`head_scene_cos` TIR .410 < RGB .468 / NIR .461。TIR head 最独立，RGB/NIR head 仍更接近；所以 TIR-TIR/TIR top-1 受益，但 NIR 没有被给出一个足够独立的 head，NIR 无专门保护/最弱，NIR→TIR 弱；TIR 的 mAP 仍受表征限制。注意两对 late <0.5，heads 在收敛，不是三个 gallery-specific 方向。

### 5.3 结论

- **不是随机**：这是“强模态（RGB）+ 被保护模态（TIR）+ 无保护中间模态（NIR）+ classifier-level GAC + paper 强 cross”的共同结果。
- **不是“只有两条普遍上涨”**：vs H014 是 9/9，vs V2 是 7/9 R1/8/9 mAP；但 **vs paper** 只有 RGB-RGB/TIR-TIR 的 R1/mAP 双涨。
- **GAC 的收益偏向 same/强模态**；cross mAP 仍需要 feature-level alignment/ranking，而不是再加 classifier。

---

## 6. 为什么 V4 vs V2 是小幅/噪声内改进

- 1000 +0.09/+0.28/+0.47/−0.21；9 对平均 +0.50/+0.12；AG mAP +0.10/+0.07；GD +0.03/−0.41/+1.11/−0.31。
- 有效 aux share 2–6%（mean ~3.35%，last-12 ~3.4%）；clip 100/114，但 V2 末期也是 100/114（pre-clip V2 2.20 vs V4 2.35），clip 不是 GAC 特有；特征几何/bank 几何与 V2 几乎一致；GAC 是弱正则。
- 单 seed；seed 噪声 mAP 0.24/R1 1.13；V4 vs V2 的 1000 差异在噪声内；9 对 +0.50 也在噪声内；不能宣布 V4 赢 V2。
- V4 vs V1：mAP/R1 +0.19/+0.29，也在噪声内；R5/R10 −0.19/−0.34；AG G→A R1 −0.83；TIR→TIR −0.33/−0.14。V1 own 1.0 的 tail/AG 优势没有被完全保住。
- V4 vs V3：+0.38/+0.84/+0.81/+0.30；R1 仍 <1.13，但四项同向；V4 比 V3 更可信；GD R5/R10 低于 V3。9 对 7/9 R1、9/9 mAP 正；但 **TIR→RGB +0.09、TIR→NIR −0.05 vs V3**，V4 没有在 TIR cross 上超过 V3；V4 vs V3 的增益主要是 same/NIR-RGB/RGB-NIR/RGB-RGB，且约 2/3 是 V2 复位。
- **epoch 稳健性（12 次 eval）**：V4−V2 mAP 12/12 正、R1 11/12、R5 7/12、R10 6/12；V4−V1 mAP 11/12、R1 8/12、R5 1/12、R10 2/12；V4−V3 48/48。V4 vs V2 是小而一致的 mAP/R1 offset，但 R5/R10 后期转负；V4 vs V1 是 mAP/R1 vs R5/R10 的 trade；12 checkpoints 相关，不是独立 seeds。
- **V4 是 trade，不是全面支配**：mAP/R1/same/cross/AG mAP/GD mAP 正；R10/GD R1/R5/R10/AG A→G R10 负。
- **分解**：V4 vs V2 = **GAC 的净效果**（V2 复位后只加 GAC）；V4 vs V3 = **V2 复位 + GAC**。V2 vs V3 本身是 +0.29/+0.56 overall、+0.54/+0.58 same、+0.02/+0.17 cross；V4 vs V3 的 +0.38/+0.84 overall、+1.14/+0.75 same、+0.47/+0.27 cross 主要来自 V2 复位，GAC 只额外加 +0.09/+0.28 overall、+0.60/+0.17 same、+0.45/+0.10 cross。**不要把 V4 vs V3 的全部增益归给 GAC。**

---

## 7. 为什么 GD mAP 最高但 R1/R5/R10 混合

- GD pair 平均 V4 最高（19.82/16.81）；但 overall R1 31.20 < V2 31.61。
- GD 的 R1 损失在 **mixed-gallery**：GAC 改善 pair-space（RGB→NIR +2.44、RGB→TIR +2.13、TIR→RGB +0.61），但 mixed-gallery 的 cross hard negative/排序没有机制；pair 改善不足以抵消 mixed R1。
- GD R5/R10 低于 V3：V3 的 cross/global 拉宽 coverage/tail；V4 的 GAC 更 top-1-heavy；GD mAP 最高，但 R5/R10 让位。
- GD NIR→TIR R1 −2.13、TIR→TIR −1.22 vs V2：TIR same/cross 的 GD 子集不稳；GD **984 valid query**（399/模态是 same-camera 排除前），1pp≈9.84 query；GD R1 −0.41≈4 query、R5 +1.11≈11、R10 −0.31≈3；需 seeds。
- paper+1 GD：只有 R10 过；mAP/R1/R5 差 0.56/1.70/0.07；GD 是最大硬缺口。

---

## 8. 为什么 AG 改善，mAP 只差 0.30（≈4–6 query，near noise）

- AG 是 all-modality；GAC 的额外 classifier 改善模态特征；V4 A→G 12.20/35.26/60.07、G→A 14.00/31.83/57.97；AG mAP 四方案最高。
- A→G R1 +0.52 vs V2、mAP +0.10；G→A R1 −0.31、mAP +0.07；A→G R10 −0.82，说明 top-1/mid 改善、tail 有代价。
- 对 paper+1：R1/R10 都过，mAP 只差 0.30/0.30；A→G 1350/G→A 1932 valid query，0.30%≈4–6 query，**near noise**；V4 mAP 比 V2 只 +0.10/+0.07（≈1–1.4 query）。**不要把 0.30 当成精确平台 residual；平台残差主要在 A→G R10（≈11 query），不在 mAP。**
- 这支持“模态改善会外溢到 AG”的大方向；但 **不是自然全过**：A→G R10 和 G→A R1 有 trade，平台机制仍是后续且低优先级。

---

## 9. 机制设计不完善 vs 代码修改不彻底

### 9.1 训练代码：正确

- aux_heads 注册/初始化/loss/target/DDP/AMP/λ=0/推理/resume 都正确；P0=0/P1=0；没有训练 bug。**但交付日志没有 λ=0 smoke**（第一次 run 在 epoch1 崩溃前无 checkpoint）；“λ=0 严格 V2”是代码验证，不是本次实验实证。
- `GAC heads initialized from P_scene: M=3, P_id fallback rows=0`；`loss_aux` 5.43→0.11；heads 远离 P_scene；说明机制确实在跑。
- `mode≥1` 与 pre-BN 一致；test 不受影响；resume 已加固。

### 9.2 诊断/归因代码：不完整

- 设计 §6.1 要求 **per-head CE/acc/grad share**；代码只落了 `loss_aux` 总量、`share_aux` 总量、`head_scene/head_id/head_pair cos`，**没有 per-head CE/acc、没有 per-head 梯度份额**。
- 设计提到 shared W_aux 是 GAC 的 collapse case/control；实验没有跑 **matched shared W_aux control**，也没有 **head ablation**（freeze/zero cross heads）。
- 因此无法严格区分：
  - 收益来自“gallery conditioning”还是“多一个 pre-BN classifier”；
  - home head 与两个 cross head 各贡献多少；
  - cross mAP 的 +0.10 是 cross head 还是 home head/全局项。
- 这是 **代码/实验完整性问题**，不是训练 bug；也是“代码不够彻底”的唯一实质点。V4 的结论“GAC R1 有信号但 mAP gate fail”是 [实验支持]；“gallery-conditioned”因果是 [机制假设]，未建立。

### 9.3 机制设计：主要瓶颈

- GAC 是 classifier-level；heads 训练后丢弃；特征几何/bank 几何与 V2 几乎一致；cross mAP 需要 feature alignment/ranking，classifier 不够。
- 有效 share 2–6%（mean ~3.35%，last-12 ~3.4%）；clip 100/114，但 V2 末期也是 100/114（pre-clip V2 2.20 vs V4 2.35），clip 不是 GAC 特有；名义 cross 2/3 没有变成有效 cross 梯度（估计 effective cross ≈2.3% of total gradient）。
- 没有 modality-specific head weights（只继承 V2 own/global/cross 权重；GAC heads 等权）；NIR 没有专门保护。
- 没有 mixed-gallery/gallery-aware ranking、没有 TIR representation/translation、没有 platform term。
- **因此：训练代码正确；机制设计是主因；诊断/归因不完整是次因，不是训练 bug。**

---

## 10. V4 证明了什么、没证明什么

**证明/支持：**

- GAC heads 可以偏离 P_scene、heads 多样（head_scene .46、head_pair .756/.459/.439），不是 V3 的 same-bank 塌缩。
- V4 在 1000 overall mAP/R1、same/cross、AG mAP、GD mAP 上都是四方案最高；对 UAD H014 九对 9/9 R1/mAP 正。
- 1000 R1/R5/R10 过 paper+1；AG R1/R10 过；GD R10 过。
- V4 vs V3 在 overall/same/cross/AG 上全面正；九对 7/9 R1、9/9 mAP 正（NIR→TIR/TIR→NIR R1 小负）；GD mAP/R1 正、R5/R10 负。

**没证明/未完成：**

- 九对 cross mAP 没有接近 paper+1：cross 14.17/9.26 vs 15.26/10.50；vs paper 14.26/9.50 也略低。
- 九对 paper+1 只有 RGB→RGB 完整过、TIR→TIR R1 过；平均差 0.68/0.98。
- GD R1 差 1.70、R5 差 0.07、mAP 差 0.56；AG mAP 差 0.30。
- V4 vs V2/V1 的 1000 差异在 seed 噪声内；不能宣布 V4 全面胜。
- “gallery conditioning”的因果贡献未严格识别（无 control/ablation/per-head grad）。
- 2337 未跑。

---

## 11. 下一步（只给方向）

1. **本轮不跑 seed 5678**：按用户口径，V1–V4 既往实验都在同一 seed1234，方法间按 same-seed paired 对比；用效应量、12-checkpoint 一致性、query-level bootstrap 判强弱；2 seeds 后置，不作为本轮 gate 前置。
2. **补 λ=0 same-code control**：V2 不是同代码 λ=0 anchor；没有它，V4−V2 混入 GAC + run/code drift。再补 per-head CE/acc/grad + shared W_aux control；否则 gallery-conditioned 归因不成立。
3. **GAC 若继续**：唯一预注册 retry 是 λ 0.1→0.15、home 0.5/cross 1.0（可选 orthogonality）；预期 cross mAP +0.1–0.3，不是 +1.24；不要加更多 heads。
4. **九对 cross/paper+1**：需要 feature-level alignment/translation 或 pair-space ranking/AP；TIR cross mAP、NIR→NIR、NIR→TIR 是主缺口；不要继续在 class-mean bank 上加 CE。
5. **GD**：mAP 只是 +0.03 tie；R1/R5/R10 混合；mixed-gallery ranking/score 是 GD R1 的主因；GD 984 valid，1pp≈9.84 query，需 seeds。
6. **AG**：mAP 差 0.30 ≈4–6 query，near noise；平台机制低优先级，监控 A→G R10/G→A R1 trade。
7. **2337**：先做缺模态预检（PKM sampler `divmod(need,0)`），再作为 Tier2 generalization check；不要用 2337 救 1000 mAP。

---

## 12. 一句话

**V4 的训练代码是正确的；GAC 是“同一 pre-BN 特征上的额外 classifier family”，heads 部分独立但 late 趋于收敛（head_pair .699/.329/.308→.756/.459/.439，两对 <0.5），有效 share 约 3.4%，bank/feature 几何仍是 V2；它买到了 R1-heavy 的正则（cross +0.45 R1/+0.10 mAP、same +0.60/+0.17），但 V4 预注册机制 gate 在 mAP 上失败（cross 9.26<9.46、九对 12.88<12.96），gallery-conditioned 因果未建立（无 per-head CE/grad、无 λ=0/shared control）。V4 是 UAD paper 水平（九对 +0.32/+0.02 vs paper），不是 paper+1（−0.68/−0.98；cross −1.09/−1.24；GD R1 −1.70）。V4 vs V2/V1 在 seed 噪声内；V4 vs V3 清楚但约 2/3 是 V2 复位。下一步先 2 seeds + λ=0 control；若继续分类器族，最多 λ/head-weight retry（+0.1–0.3 cross mAP）；paper+1 需要 TIR representation/ranking 或 mixed-gallery calibration。**
