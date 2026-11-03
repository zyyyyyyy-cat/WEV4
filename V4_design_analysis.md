# WEV4 / GAC：V2 复位 + Gallery-Conditioned Auxiliary Classifier

> 修订记录（2026-11-01）：初版主方案 CMFA（跨模态正样本对齐）经独立机制评审后降为对照/拒绝项；主方案改为 **GAC（Gallery-Conditioned Auxiliary Classifier，gallery 条件辅助分类器）**。理由见 §2。
> 执行口径更新（2026-11-02）：本轮指标强弱分析按用户要求**只使用 seed1234 单 seed**；V1–V4 既往实验均为同一 seed，方法间按 same-seed paired 对比；2 seeds 后置，不作为本轮 gate 前置。
> 目标：九种模态检索协议（论文 Table 6）第一，尤其 cross 6 对；AG A→G / G→A 第二；V2 复位；训练-only；推理不变。
> 数据：V2 九对平均 19.47/12.76 vs paper+1 20.65/13.86；cross 6 对 13.72/9.16 vs 15.26/10.50。
> 标记：[代码事实] / [实验支持] / [机制假设]。单 seed 不裁决 <0.5pp。

---

## 0. 一句话

**V4 = V2 复位（P_id/P_scene 0.75/0.25，去掉 V3 gallery cross）+ GAC：三个 gallery 条件的可学习分类头，从 P_scene 初始化，每个头都作用在所有 query 模态上，global 500 类 CE，训练-only，推理不变；λ_aux=0.1，一个机制、一个 loss、一个 λ；用 head-cos 诊断判断分类器族是否还有新方向。**

---

## 1. 目标与基线

### 1.1 九对 Table 6 的缺口

| Pair | V2 R1/mAP | paper+1 R1/mAP | V2−paper+1 | 备注 |
|---|---:|---:|---:|---|
| RGB→RGB | 42.20/27.52 | 41.47/27.83 | +0.73/−0.31 | R1 已过，只差 mAP |
| RGB→NIR | 27.12/16.50 | 27.56/17.52 | −0.44/−1.02 | NIR |
| RGB→TIR | 7.26/5.71 | 9.10/7.00 | −1.84/−1.29 | TIR |
| NIR→RGB | 25.25/16.11 | 26.53/17.58 | −1.28/−1.47 | NIR↔RGB |
| NIR→NIR | 29.60/17.80 | 30.79/18.93 | −1.19/−1.13 | NIR same |
| NIR→TIR | 7.03/5.29 | 8.78/6.87 | −1.75/−1.58 | TIR |
| TIR→RGB | 7.59/5.98 | 10.04/7.32 | **−2.45/−1.34** | 最大 R1 缺口 |
| TIR→NIR | 8.06/5.39 | 9.57/6.69 | −1.51/−1.30 | TIR |
| TIR→TIR | 21.12/14.55 | 21.98/15.00 | −0.86/−0.45 | TIR same |
| **平均** | **19.47/12.76** | **20.65/13.86** | **−1.18/−1.10** | |

- V2 九对 **8/9 R1、9/9 mAP 低于 paper+1**；只有 RGB→RGB R1 已过。
- **cross 6 对平均 13.72/9.16 vs 15.26/10.50，gap −1.54/−1.34**；same 3 对 30.97/19.96 vs 31.41/20.59，gap −0.44/−0.63。**cross 是主缺口，约 same 的 3 倍。**
- **vs UAD 论文**，V2 cross 只低 −0.54/−0.34；**1.54/1.34 里大部分是 paper+1 的目标差，不是相对 UAD 的崩塌**。真正的 deficit 是 TIR 主导：TIR→RGB −2.45、RGB→TIR −1.84、NIR→TIR −1.75、TIR→NIR −1.51；NIR→RGB −1.28、NIR→NIR −1.19。这是 **TIR 表征 + 排序** 问题，不是缺一个共享锚点。
- 论文 Table 6 的九对以 R1/mAP 为主；R5/R10 逐对日志未打印，可从 `pair_rank_query.csv` 的 rank 离线计算，不改训练/推理。

### 1.2 为什么 V2 复位

- V2 是 overall/cross/GD 最强基线：1000 11.47/30.52/47.51/56.14，cross 13.72/9.16，GD 14.91/31.61/44.82/54.17。
- V3 的 gallery cross 是从 V2 的 own/global 预算里切出来的，三层同 bank（P_id=mean(P_scene) cos 0.999；cross 正样本也是 class mean；grad cos own-cross 0.714 / global-cross 0.908），结果是零和。
- V4 **不重分配 V2 的 own/global**；GAC 是**训练-only 的加法项**，不是把 0.75/0.25 改成 0.60/0.20/0.20。

### 1.3 目标层级

| 层级 | 目标 |
|---|---|
| 机制信号（本轮 go/no-go） | cross 6 对均值 ≥ V2+0.4 R1/+0.3 mAP；九对均值 ≥ V2+0.2/+0.2 |
| 单对护栏 | 无 pair 低于 V2−0.5；NIR→NIR、NIR→RGB、TIR→RGB ≥ V2−0.3 |
| same/AG/GD 护栏 | same 均值 ≥ V2−0.3；AG mAP ≥ V2−0.3；GD mAP ≥ V2−0.3；overall mAP ≥ V2；R5/R10 ≥ V2−0.5 |
| Paper+1（Tier2） | 九对平均 20.65/13.86；cross 15.26/10.50；不作为单机制 go/no-go |
| 2337 | 等 cross/九对 signal + 2 seeds 后再跑 |

---

## 2. 从 CMFA 到 GAC：独立评审的裁决

### 2.1 初版 CMFA 为什么被否

初版主方案是 **CMFA**：pre-BN 上按 identity 把不同模态的 modality mean 拉近。独立机制评审给出的反对理由：

1. **pair cos 已经很高**：V2 RGB-NIR 0.918、RGB-TIR 0.803、NIR-TIR 0.811；再对齐容易过对齐/压掉模态细节。V3 的 pair cos 比 V2 更高（0.926/0.819/0.826），但 NIR same tail 反而更差（NIR→NIR −0.93、>10 +27）；这是“对齐≠泛化”的直接证据。
2. **ProCA/GPD global 已经在做对齐**：CMFA 与它们同方向，可能冗余；预期 cross 只有 0 到 +0.2，不是 +1.5。
3. **CMFA 只用正样本**：不覆盖跨模态负样本/排序；cross 的 R1/AP trade、TIR cross mAP 跌，说明瓶颈包含负样本/排序，不只是正样本距离。
4. **cross 的真正 deficit 是 TIR 表征 + 排序**：TIR→RGB 7.59/5.98，即使 UAD 论文也只有 9.04/6.32；CMFA 不创造 TIR 身份细节，也不优化 AP。

**结论：CMFA 是低风险但低杠杆的“对齐”假设，作为对照可以，不作为 V4 主方案。**

### 2.2 候选机制比较

| 候选 | 机制 | 对 cross | 对 same | 主要冲突 | 推理 | 裁决 |
|---|---|---|---|---|---|---|
| V3 gallery cross | 对 P_scene 另一列的 class-mean CE | 仅 TIR query R1，mAP 不涨 | 被挤 | 同 bank/零和 | 不变 | 拒绝 |
| shared W_aux | 一个共享 pre-BN 分类头 | +0.1~0.4，和 CE/GPD global 高冗余 | +0.2~0.5 | 与 CE/global 同向 | 不变 | 低风险探针，不做主跑 |
| per-modality aux | 每模态一个头，只在 own 段训练 | −0.2~+0.2，头可发散 | +0.3~0.8 | 复制 GPD own；NIR donor | 不变 | 方向不对 |
| shared + residual | 共享头 + 每模态残差 | +0.1~0.4 | +0.3~0.6 | 共享 vs global、残差 vs own | 不变 | same 回退时再用 |
| **GAC（主方案）** | **3 个 gallery 条件头，每个头作用在所有 query 模态，global 500 类 CE** | **+0.3~0.8（头多样时）** | +0.1~0.4 | home head vs own；cross head vs global；头塌缩/发散 | 不变 | **主方案** |
| pair-space AP/ranking | 对 query–prototype 相似度做 AP/ranking | +0.2~0.6 | +0.1 | 同 bank；score-scale | 不变 | GAC 后 V5 |
| modality adapter | 每模态低秩 adapter | +0.5~1.5 潜力 | +0.5~1.0 | 改特征、KGL 历史 | 改变 | GAC/ranking 失败后 |
| feature alignment/translation | 对齐/翻译模态特征 | 0~+0.2，过对齐 | −/0 | 与 ProCA/global 冗余；塌缩细节 | 可能改变 | **不推荐（CMFA 即此类）** |
| platform prototype | 平台原型/采样 | 0~+0.2（Table 6） | 平台内 | 碎片化 bank；AG 专用 | 不变 | AG 独立轮次 |

### 2.3 为什么是 GAC

- **cross 需要的是 gallery 条件的判别边界，不是一个共享锚点**。GAC 的头 `W_m` 对应 gallery 模态 m；对每个 query，三个头都算 CE；home 头保同模态，cross 头直接监督跨模态。
- **global 500 类负样本**：每个 CE 都用全部 C 个身份权重，不像 batch-local ranking 只有 P=16 个 identity/15 个负样本；保留 GPD 成功的全局判别。
- **learned，不是 class mean**：V3 的 cross 正样本是 `P_scene[c,m']` 固定 class mean；GAC 的正样本是 `W_m'[y]` 可学习权重，可以和 class mean 分开。`cos(W_m, P_scene[:,m])` 直接检验它是否逃出 V3 的同 bank 陷阱。
- **一个机制，一个 loss，一个 λ**：三个头是同一机制的条件化，不是三个模块；不做 residual/adapter/ranking/platform。
- **GAC 是 shared W_aux 的严格推广**：三个头相等时就是 shared W_aux；头发散时是 cross 监督。一次 run 同时回答“shared 够不够”和“cross 条件化有没有新方向”。

### 2.4 独立代码审计：aux-head 路线的 P0/P1

已对 `WEV2/WHU` 做只读审计。GAC 属于 aux-head 路线，必须解决：

**P0：**

- **P0-1**：`aux_heads` 必须在 DDP 包装的 `forward` 内计算，不能在 processor 里直接调 `model.module.aux_heads(...)`；否则 `find_unused_parameters=True` 下梯度不同步/“ready twice”。
- **P0-2**：不能无条件改 mode-0 返回元组；`diagnose_gpd_gradients` 在 `processor.py:451` 按 3 元组解包。用 `return_aux=False` 默认参数。
- **P0-3**：`aux_heads` 必须无条件注册、名字稳定；`test.py:55-56` 的 `load_param` 按 checkpoint key 索引，条件创建/改名会 `KeyError`。
- **P0-4**：mode ≥1 的推理路径必须逐字节不变；aux logits 只能是独立 tensor，不能改 `global_feat`/`feat`，特征维仍是 768。

**P1（摘要）：**

- 注册在 bottleneck 之后；用 `weights_init_classifier`/`bias=False`；名字 `aux_heads`，避开 `classifier`/`arcface`/`bias` 的 LR 规则。
- `load_param` 跳过 `aux_heads`；避免 V2/V4 checkpoint 的 key/形状问题。
- `λ_aux=0` 必须 gate 掉 forward/loss；heads 以 `torch.zeros`/no-RNG 构造（不需要 `fork_rng`），避免 weight decay/RNG 偏移。
- 新 config key 默认 0.0；独立 `aux_meter`；`processor.py:794-799` 加 loss；optimizer/scheduler 自动包含新参数。
- AMP：`global_feat.float()`、heads/logits fp32 CE；DDP 用 `find_unused_parameters=True`。
- 诊断 `diagnose_gpd_gradients` 要加 aux share/cos，且不能改变原 3 元组调用。

**含义**：GAC 比 CMFA 多 4 个 P0 + 约 10 个 P1；这是“cross 条件化”应付的工程成本。CMFA 虽简单，但机制杠杆不对；不能因为好实现就选错机制。

### 2.5 最后独立审查（2026-11-01）

**裁决：approve with conditions。** 设计可在 V2 上实现，不需要改机制；问题主要是 silent-failure 陷阱。1ep smoke 需先落 P0-1..P0-6；120ep 需 P1-1/P1-2/P1-4；P1-3 在 gate 判定前解决。

| ID | 问题 | 修复 |
|---|---|---|
| **P0-1** | heads 在 `make_optimizer` 之后创建 -> 不被优化，GAC 退化为固定 P_scene cross | 在 `build_transformer.__init__` 注册 zeros/no-RNG placeholder，warm start 后 fill |
| **P0-2** | 只在 rank0 init -> DDP 各 rank W 不同 | bank broadcast 后所有 rank 从 P_scene 填；或显式 `dist.broadcast(aux_heads, src=0)` |
| **P0-3** | `target_rep.repeat(M)` 与 `[N,M,C]` flatten 不匹配 -> 标签错 | `target_rep.unsqueeze(1).expand(-1,M).reshape(-1)` 或 per-head loop |
| **P0-4** | heads 在 DDP forward 外 / `return_aux` 破坏 3 元组 / mode≥1 改变 | forward 内算；`return_aux=False` 默认；mode≥1 不变 |
| **P0-5** | 用 post-BN `feat` 或 `detach(global_feat)` -> 机制 test-inert | 用 pre-BN `global_feat`；**不要 detach** |
| **P0-6** | λ=0 未 gate；head 创建消耗 RNG | λ>0 才 forward/loss；heads 零初始化/不消耗 RNG；fill 在 warm-start RNG 之后 |

**P1：**

- **P1-1**：AMP fp32 不是自动；head matmul+CE 用 `autocast(enabled=False)` 或放在 autocast 外，否则是 fp16 logits。
- **P1-2**：命名 `aux_heads`；`load_param` 跳过；无条件注册才是硬要求。
- **P1-3**：gate +0.4 低于 <0.5 不裁决；V2 只有 seed1234，V4 2-seed 不 paired。补 V2 seed5678 或把 +0.4 当 provisional；query-level bootstrap 不解决 seed 方差。
- **P1-4**：warm start off / bank all-zero -> heads 相同 -> shared W_aux；λ>0 且 bank 未 warm start 直接 raise。
- **P1-5**：不要在 epoch1 看 aux CE 决定 2× LR；预注册 LR；pairwise head cos 用 per-class mean cos，不是 flatten。
- **P1-6**：单独 aux meter；aux loss 在 `scaler.scale` 前；heads 在 `clip_grad_norm_` 内；V2 clip 100/114，监控饱和。

**设计一致性：**

- **Table 6 是 WHU-MARS-1000**；2337 不是 Table 6 的前置。
- 等权 (n,m) = 9 种 pair 类型等权：cross 6/9、same 3/9；home 1/3 与 CE/GPD own 冗余，有效 cross 权重约 **2/3 λ**；预期 cross 幅度要打折。
- “gallery-conditioned” 只是 **init-anchored**；loss 对三个 head 置换对称，没有 routing 项。若 W 不偏离 P_scene，GAC cross 在功能上接近 V3 `GPD_TPD` cross + additive budget + home；若塌缩，就是 shared W_aux。`cos(W_m,P_scene)` 只能分类结果，不能因果归因 gallery conditioning；要因果归因需 matched shared W_aux control（§5.1 提到但 §9 没有）。paper claim 限定为 “aux classifier family”，除非跑 control。
- paper+1 是 Tier2；+0.4 cross gate 不等于 Table 6 paper+1。
- 约束满足；head 数 = `len(cfg.DATASETS.MODALITIES)`/`GPD_SUB_CENTERS`，不硬编码 3。

---

## 3. GAC 设计

### 3.1 结构

- 头 `W ∈ R^{M×C×D}`（M=len(modalities)=3，**不硬编码**；head 数 = `len(cfg.DATASETS.MODALITIES)`/`GPD_SUB_CENTERS`），无 bias；每个头每次 forward 做 L2 normalize（cosine classifier）。
- **无条件注册**：在 `build_transformer.__init__` 注册 `nn.Parameter(torch.zeros(M,C,D))`（zero/no-RNG placeholder），保证 `make_optimizer`/DDP/state_dict 在 warm start 之前就看见它；warm start 后再 fill。若在 warm start 后才创建，参数不会被 optimizer 收录，GAC 退化为固定 P_scene cross（P0-1）。
- 初始化：V2 warm start + bank broadcast 之后，**若 `AUX_WEIGHT>0`，所有 rank** 上 `W[m] = L2(P_scene[:,m])`；`P_scene` 是 `[C,K,D]` 的第 m 个模态 class mean。若 bank 有零行，用 `P_id/g_cent` fallback；若 bank 全零/未 warm start，直接 raise（P0-2/P1-4）。`AUX_WEIGHT=0` 时保持 zero/不初始化，作为 V2 anchor。
- `f = L2(global_feat.float())`，`global_feat` 是 pre-BN CLS `[M*B,D]`；**不要 detach global_feat**（detach 会让 heads 训练但检索特征不变，机制 test-inert，P0-5）。
- `logits[n,m,:] = (f[n] · W[m]) / τ`，`τ=0.03`；`[M*B, M, C]`。
- **AMP**：head matmul+CE 用 `autocast(enabled=False)` 或放在 autocast 外；`.float()` 不会自动阻止 autocast 把 matmul 转 fp16（P1-1）。
- 对每个 query n 和每个头 m：`CE(logits[n,m,:], y[n])`；`L_aux = mean_{n,m} CE`。
- 实现细节：`logits` 是 `[N,M,C]`（`N=M*B`）。可以按 head 循环 `mean_m CE(logits[:,m,:], target_rep)`，或 flatten 配 `target_rep.unsqueeze(1).expand(-1,M).reshape(-1)`（等价 `repeat_interleave(M)`）；**不要用 `target_rep.repeat(M)`**（那是 head-major，和 `[N,M]` 的 flatten 顺序不一致，P0-3）。
- 总损失：`CE + WRT + 0.01 ProCA + 1.0 GPD + λ_aux L_aux`。
- `λ_aux=0.1`；epoch 1 0.05 → epoch 5 0.10，之后 constant。
- 训练-only；推理 pre-BN CLS + L2 + 欧氏；不加载 heads/bank。

### 3.2 为什么 gallery-conditioned

- 头 m 代表 gallery 模态 m；对 query 模态 q：
  - m=q 是 home 头，保同模态判别；
  - m≠q 是 cross 头，直接把 query 特征推到 gallery 模态 m 的身份边界。
- 三个头都作用在所有 query 上，所以每个头都必须跨模态兼容；这避免了 per-modality 头“各管一段、互相发散”的问题。
- 每个头都是 global C 类 CE；负样本是全部身份，不是 batch 内 15 个负样本。

### 3.3 为什么 learned 头不是 V3 的同 bank

- V3：`L_cross = CE(cos(f, P_scene[c,m'])/τ)`；正样本是固定 class mean，和 own/global 同一个 bank；梯度 cos 0.714/0.908。
- GAC：`L_cross = CE(cos(f, W_m'[c])/τ)`；`W_m'` 是可学习权重，初始化为 `P_scene[:,m']`，但可以偏离。
- 诊断 `cos(W_m, P_scene[:,m])` 和 pairwise head cos：
  - 若 pairwise cos >0.99、`cos(W_m,P_scene)` >0.99：GAC 塌缩为 shared/V3-like；分类器族没有新方向。
  - 若 pairwise cos 0.5–0.95、`cos(W_m,P_scene)` 下降：GAC 真的学出了 gallery 条件方向；cross 才有新信息。
- 这就是为什么 GAC 是“可证伪”的：head 诊断直接回答“分类器族还有没有新方向”，不用猜。

### 3.4 约束（保证机制发挥预想功能）

1. **heads 在 DDP forward 内**；不能 processor 直接调 submodule。
2. **无条件注册**；`load_param` 跳过 `aux_heads`；mode ≥1 不变。
3. **cosine + L2 + τ=0.03**；不要用 raw 特征点积（norm shortcut）。
4. **初始化自 P_scene**；不要随机初始化（早期 CE 会 spike，且消耗 RNG）。
5. **λ=0 gate**；不计算 forward/loss；保持 V2 anchor。
6. **heads 以 `torch.zeros`/no-RNG 构造**；不需要 `fork_rng`；保持数据/增强流与 V2 一致。
7. **一个 λ、一个 loss**；不做 residual/adapter/ranking/platform。
8. **head-cos 诊断**；塌缩/发散是判定的一部分，不是训练失败。
9. **前置守卫**：GAC 要求 `GPD_TYPE='hierarchical'`、`GPD_WARMSTART=True`、`GPD_SUB_CENTERS==M`、`TEST.NECK_FEAT='before'`；否则 P_scene 不存在/未初始化/列不对齐，head 初始化无意义，直接 raise。V2 配置已满足；不要在没有 warm start 的变体上静默跑。若 bank 全零或 `W[m]` 全零，heads 三个相同、loss 置换对称 -> GAC 就是 shared W_aux；λ>0 时直接 raise。head 数 = M，不硬编码 3。

---

## 4. 模块协同与冲突

| 模块 | 方向 | 与 GAC 的关系 | 冲突/冗余 | 约束 |
|---|---|---|---|---|
| **BNNeck CE**（post-BN, 1.0） | 共享身份分类 | GAC 是 pre-BN 的 gallery 条件分类；层不同 | 高冗余（同 function class）；GAC cross 头加 pair routing | 监控 head-cos；若 GAC 只复制 CE，cross 不涨 |
| **WRT**（pre-BN, 1.0） | 同相机负样本 + 同 identity 正样本 | GAC 提供 global 500 类负样本；WRT 提供同相机局部负样本 | budget 竞争；GAC 可能压缩 WRT 的正样本多样性 | λ=0.1；监控 clip/WRT loss |
| **ProCA**（pre-BN, 0.01） | modality center → g_cent | GAC cross 头拉跨模态；ProCA 保 modality center | 方向对抗；GAC 可能让 P_scene 列更近 | PCA loss ≥V2 90%；own_ratio/pair cos |
| **GPD own**（pre-BN, 0.75） | modality class mean | GAC home 头复制 own；cross 头加新方向 | home 头冗余；cross 头与 own 对抗 | own 0.75 主导；λ=0.1；same 护栏 |
| **GPD global**（pre-BN, 0.25） | P_id=mean(P_scene) | GAC cross 头与 global 都做跨模态 | 可能冗余；head-cos 高则 GAC 无新方向 | cross 信号；head-cos/`cos(W,P_id)` |

### 4.1 监督是否足够

| 目标 | 监督 | 是否足够 |
|---|---|---|
| 同模态判别 | CE + WRT + GPD own + GAC home | 足够 |
| 跨模态正样本 | GAC cross 头 | 直接覆盖 |
| 跨模态负样本 | GAC 每个头的 global 500 类 CE | class-level 覆盖；instance-level hard negative 未覆盖 |
| 同相机负样本 | WRT | 覆盖 |
| 模态中心/多样性 | ProCA + GPD own | 覆盖，但与 GAC 对抗 |
| 平台 | 无 | 未覆盖；AG 副产物 |

**结论**：GAC 补的是“gallery 条件的跨模态判别”，比 CMFA 多了 **cross 负样本/global 500 类** 这一块；但 instance-level hard negative 仍未覆盖。如果 cross R1 涨、mAP 不涨，下一步是在 GAC logits 上做 AP/ranking（V5），不是回到 P_scene 原型。

### 4.2 主要 failure modes

1. **冗余/空转**（shared W_aux 最可能）：`cos(g_aux,g_id/g_scene)` 高、head pairwise cos 高、cross <0.2。说明分类器族没有新方向。
2. **过锐化/NIR test tail**：train geometry 更好、per-pair train acc 更高，但 NIR→NIR/NIR→RGB test tail 更差。V3 已经出现过；监控 train vs test class-mean geometry 和 rank>10。
3. **TIR 过对齐/donor**：GAC 把 TIR 拉向 RGB 主导的边界；TIR→TIR mAP 跌、TIR cross R1 涨 mAP 跌（V3 模式）。TIR 保护不能只靠 λ；要靠 head-cos/own 护栏。
4. **Norm shortcut**：heads 必须 cosine/L2；不要 raw 特征点积。
5. **Budget dilution**：aux share 5–10% 会稀释 own/WRT；same/AG/R5 可能回退。监控 last 10–12 epoch norm-of-sum，不用单 batch。
6. **Head collapse**：pairwise cos >0.99；GAC 退化为 shared W_aux。这是诊断结果：分类器族没有新方向。
7. **Head divergence**：pairwise cos <0.5、per-head CE/acc 不平衡、cross mAP 跌；per-modality 发散风险从后门回来。用 P_scene 初始化、等权、per-head 梯度诊断约束。
8. **DDP/AMP/clip**：heads 在 DDP forward 内；fp32 CE；V2 末期 pre-clip 1.73/100 clips；加 aux 会更高；不改 max_norm，降 λ/延长 ramp。

### 4.3 冲突约束

- `share_aux` 目标 5–10%；>15% 降 λ；<3% 视为 inert。
- `cos(g_aux, g_id/g_scene)` >0.8 视为高冗余；cross 不涨则停。
- pairwise head cos：>0.99 塌缩；<0.5 发散；0.5–0.95 是“有新方向”的区间。
- `PCA loss` ≥V2 90%；same/AG 回退 >0.3 触发降 λ；>0.5 停。
- V2 末期 clip 100/114；GAC 若 clip 率接近饱和，降 λ/延长 ramp，不改 max_norm。

---

## 5. 训练与优化

### 5.1 λ 与 ramp

- GAC：`λ_aux=0.1`；epoch 1 `0.05` → epoch 5 `0.10`，之后 constant。
- heads 从 P_scene 初始化，初始 aux CE 与 GPD 同量级，不会像随机头一样 spike。
- 不做 0.05/0.15 多臂；如果 0.1 有信号但幅度小，下一轮同一机制内做一次 λ/head-weight refinement。
- shared W_aux（仅作为塌缩对照，不主跑）：λ=0.15，一个头初始化自 P_id；不与 GAC 同跑。

### 5.2 LR/WD/初始化

- 沿用 V2：SGD，LR 0.032，batch 256，120 epoch，cosine+5 warmup，WD 1e-4，clip 1.0。
- heads 用 backbone 同 LR；**预注册，不在 epoch1 看 aux CE 改 LR**（P1-5）。若需要 2× LR，作为下一次独立 run 的预注册变体，不在同一 run 中途改。
- heads 无 bias；WD 0 或 1e-4；cosine 每次 forward L2 normalize。
- 初始化在 warm start 之后：`W[m] = L2(P_scene[:,m])`；不消耗 RNG。
- **必须在 bank broadcast 之后、所有 rank 上都执行 copy**（`model_meta.aux_heads[m].data.copy_(...)`）；不能只在 main rank 初始化，否则 DDP 各 rank 的头不一致，梯度 all-reduce 也救不回来。
- 若 train aux acc >99.5% 且 NIR tail 变差：加 label smoothing 或降 λ，不同时做。

### 5.3 DDP/AMP

- heads 在 model 内（`nn.Parameter` [M,C,D] 或 M 个 `nn.Linear(bias=False)`；M=len(modalities)，不硬编码 3）；DDP all-reduce。
- `find_unused_parameters=True` 已有；λ=0 时 heads unused，和 `base.fc` 一样被处理。
- `f = F.normalize(global_feat.float(), dim=1)`；head matmul+CE 用 `autocast(enabled=False)` 或放在 autocast 外，否则仍是 fp16 logits；GradScaler 监控 overflow。
- test/do_inference 不调用 heads；mode ≥1 不变；checkpoint 含 heads，test 忽略/加载均可。

### 5.4 V2 anchor（λ=0）

- 1ep smoke：`λ=0` 不计算 aux forward/loss；heads 保持 zero/不初始化（代码 `aux_target=0` 不调用 `initialize_gac_heads`），不参与；total/per-term loss、bank counts、warm start 与 V2 在 float 误差内一致。
- heads 以 `torch.zeros` 构造（no-RNG，无需 `fork_rng`），不改变数据/增强流。
- 120ep λ=0 不需要重跑；用现有 V2 seed-1234 作 anchor，记录代码差异。
- checkpoint 会多 `aux_heads` key；test 跳过；不影响 V2 指标。

### 5.5 统计

- seeds 1234 + 5678；V2 seed 1234 已有；预算允许补 V2 seed 5678 做 paired。
- 报告 mean±std 和 per-pair delta；<0.3 R1/<0.2 mAP 用 paired bootstrap over query IDs；单 seed <0.5pp 不裁决。
- **P1-3**：gate 的 +0.4 R1 低于 <0.5pp 不裁决线，且 V2 只有 seed1234；若不补 V2 seed5678，+0.4 只作 provisional，不作最终 pass/fail。query-level bootstrap 只解决 query 采样，不解决 seed/训练方差。
- 不拿单 V4 seed 对单 V2 seed 宣布 paper+1 失败；历史 seed 噪声 mAP 0.24/R1 1.13。

---

## 6. 诊断

- `g_aux`、`share_aux`（norm-of-sum）、与 `g_base/g_id/g_scene/g_pca` 的 cos；WRT 若可分离也报。
- per-head CE/acc/grad share；home/cross accuracy；logit pos/neg/entropy。
- pairwise head cos（**per-class mean cos**，不是 flatten-matrix cos；与 P_scene pair cos 同口径）；`cos(W_m, P_scene[:,m'])`、`cos(W_m, P_id)`；这是 GAC 的核心诊断。
- `P_scene` pair cos、own_ratio、usage、feature norm、clip count；last 10–12 epoch mean。
- test query/gallery class-mean geometry；train geometry 不能解释 NIR test tail。
- 九对 R1/mAP/mINP、rank 2–5/6–10/>10；R5/R10 从 `pair_rank_query.csv` 离线；same/cross、AG/GD、1000。
- GAC 诊断单独 `diagnose_gac_gradients`，内部 `model(..., return_aux=True)`；不改变 `diagnose_gpd_gradients` 的 3 元组调用，避免 P0-2。

---

## 7. Gate 与 fallback

### 7.1 机制信号 gate（本轮 go/no-go）

| 条件 | 阈值 |
|---|---|
| cross 6 对均值 | ≥ V2+0.4 R1 / +0.3 mAP（14.12/9.46） |
| 九对均值 | ≥ V2+0.2/+0.2（19.67/12.96） |
| 单对 | 无 pair 低于 V2−0.5；NIR→NIR、NIR→RGB、TIR→RGB ≥ V2−0.3 |
| same 3 对 | ≥ V2−0.3（30.67/19.66） |
| AG/GD | AG mAP ≥ V2−0.3（11.80/13.63）；GD mAP ≥ V2−0.3（14.61） |
| overall | mAP ≥ V2（11.47）；R5/R10 ≥ V2−0.5（47.01/55.64） |
| 统计 | 2 seeds |

**统计解释（P1-3）**：+0.4 R1 低于 <0.5pp 不裁决线；若 V2 seed5678 不补，+0.4 只作 provisional，不作最终 pass/fail。最终 pass 需要 V2/V4 各 2 seeds 的 paired 比较；query-level bootstrap 不解决 seed/训练方差。

**Paper+1（Tier2）**：九对 20.65/13.86；cross 15.26/10.50。**不作为单机制 go/no-go**；V1/V2/V3 没有任何机制把九对平均移动 +1.18/+1.10，单个辅助分类器不应被卖成能关闭它。

**R5/R10**：九对逐对日志没有 R5/R10；从 `pair_rank_query.csv` 的 rank 离线算。论文 Table 6 不提供逐对 R5/R10，因此逐对 R5/R10 只做 no-regression（≥V2−0.5），不参与 paper+1 判定；整体/GD 的 R5/R10 仍按官方日志。

### 7.2 Fallback

- **头塌缩（pairwise cos >0.99）且 cross <+0.2**：分类器族没有新方向；停止 aux 调参；下一轮 feature-level ranking 或 TIR adapter。
- **头发散且 cross <+0.2**：cross 监督没有迁移；同上下一步。
- **cross +0.2~0.4**：部分信号；只做一次 retry（cross 头加权 home 0.5/cross 1.0，或 λ 0.1→0.15）；不在同一 run 加新模块。
- **cross 涨、same 回退 >0.5**：降 λ 到 0.05 或降 home 头权重；一次 retry；不加 residual。
- **cross 涨、AG/GD 回退 >0.5**：记录；平台机制独立轮次；本轮不加 platform prototype。
- **cross 涨、mAP 不涨（R1/AP trade）**：下一轮在 GAC logits 上做 AP/ranking（V5）；不回 P_scene 原型。
- **NaN/clip 饱和**：降 λ、延长 ramp；不改 max_norm；持续则停。

---

## 8. 风险

- **GAC 仍是分类器族**：如果三个头塌缩，它退化为 shared W_aux；cross 不涨；这是“分类器族没有新方向”的诊断，不是代码失败。
- **头发散**：per-modality 发散风险；用 P_scene 初始化、等权、head-cos/per-head CE 监控。
- **TIR 过对齐/donor**：GAC cross 头可能把 TIR 拉向 RGB；TIR→TIR mAP 跌；用 own/GPD 护栏和 head 诊断。
- **NIR test tail**：learned 第二分类器可能过锐化；监控 train vs test geometry、rank>10。
- **Budget dilution**：aux share 5–10%；same/AG/R5 可能回退；监控 last 10–12 epoch。
- **代码成本**：4 P0 + 10 P1；必须按 §2.4 实现，否则 DDP/checkpoint/test 风险。最后独立审查裁决 **approve with conditions**：P0-1..P0-6 先落才可 1ep smoke；P1-1/P1-2/P1-4/P1-5 在 120ep 前；P1-3 在 gate 判定前。
- **Paper 复现差**：论文 288×144 vs 代码 256×128；机制涨但离 paper+1 仍远是可能的；单独记录，不混入本轮。
- **Seed 噪声**：2 seeds；<0.5pp 不裁决。

---

## 9. 实验顺序

0. **代码实现**：先落 P0-1..P0-6（heads 注册在 optimizer 前、all-rank init、target 展开、DDP/arity/inference、pre-BN/no-detach、λ=0/RNG）；120ep 前落 P1-1/P1-2/P1-4/P1-5（AMP fp32、命名/load_param、warm-start guard、预注册 LR/head-cos 口径）。
1. 1ep smoke：`λ=0` 与 V2 loss/bank/warm start 一致；`λ=0.1` finite/NaN/DDP/clip；head 初始化自 P_scene；mode ≥1 推理不变。
2. 120ep：只跑 `λ=0.1` 一个点；seed 1234 screening。
3. 测试：1000/GD/AG/detailed-domain；九对 R1/mAP；R5/R10 离线；attribution/test geometry。
4. Gate：cross/九对/same/AG/GD；V4 seed 5678；预算允许补 V2 seed 5678 做 paired；否则 +0.4 只作 provisional。
5. 通过：2337；GD/AG 独立轮次。未通过：按 §7.2 fallback，不继续调 GAC λ/头权重。

---

## 10. 一句话

**V4 不再在 class-mean bank 里调权重，也不做“把模态拉近”的对齐；它用 V2 复位 + GAC：三个 gallery 条件的可学习头，每个头作用在所有 query 模态上，global 500 类 CE，训练-only、推理不变；head-cos 诊断直接回答分类器族还有没有新方向；cross 是主检验，paper+1 是 Tier2。**

---

## 11. 代码实现与续训（2026-11-01）

- 代码：`CC/WEV4/WHU`（从 `CC/WEV2/WHU` 复制，不覆盖 V1/V2/V3/UAD）。
- 改动文件：`model/make_model.py`、`processor/processor.py`、`config/defaults.py`、`train.py`、`configs/GAC_UAD.yml`、`README_GAC.md`；`test.py` 路径不变。
- GAC：`aux_heads [M,C,768]` 零初始化注册；warm start + bank broadcast 后所有 rank 从 `P_scene[:,m]` 填；`return_aux=True` 在 forward 内用 pre-BN `global_feat` 算 fp32 cosine logits；`L_aux` 用 `unsqueeze(1).expand(...).reshape(-1)`；λ 0.05→0.1；λ=0 严格 V2。
- 续训：每 `CHECKPOINT_PERIOD=5` 和最后一个 epoch 保存 `transformer_{epoch}.pth`（model-only）+ `checkpoint_latest.pth`（model+optimizer+scheduler+scaler+bank+RNG+epoch，原子替换）；`--resume`/`SOLVER.RESUME` 从 `epoch+1` 继续；model-only 回退到 epoch1。
- 本地 smoke：compileall、model forward aux shape、GAC loss/ramp/head init、三种 bank 的 resume save/load、full-dict `load_param` 均通过；未跑真实数据/GPU。
- 独立代码审查（含 resume）：**Approve as-is，P0=0**；P1-1（NumPy/Python RNG 未恢复）、P1-2（model-only resume 静默从零）、P1-3（bank type 未校验）已修复并复测通过。复审：**P0=0/P1=0，resume sampler 流已 faithful**。
- 续训仍按 epoch 边界；`checkpoint_latest.pth` 每 `CHECKPOINT_PERIOD=5` 写一次，崩溃最多丢 4 epoch；可设 1 更细。sampler 的 torch/cuda/numpy/python RNG 已恢复。
- 方案-代码自检：`mode≥1` 输出与 pre-BN `global_feat` 逐元素一致；`λ=0` 下 `aux_heads` 经 SGD step 不变（unused/no-WD）；文档的 λ=0/fork_rng 措辞已与代码对齐。
- **服务器首次运行修复**：`GAC diag` 日志格式串多了一个 `{:.3f}`，`head_pair_cos` list 落入浮点占位符，`GPD_DIAG_PERIOD=100` 时崩溃。已改为 `format_gac_diag`（3 个 cos 占位符 + 3 个 list `{}`），并给诊断日志加 try/except；全 `.py` AST format 检查无其他 mismatch；实际 diagnostic 路径（小模型+bank）已跑通。1ep smoke 需 `MODEL.GPD_DIAG_PERIOD 1`。
- **PyTorch 1.12.1 + CUDA 11.3**：代码只用 >=1.12 API；`torch.load(..., weights_only=False)` 有 try/except；`torch.cuda.set_rng_state` 使用 CPU ByteTensor（不 `.to('cuda')`）；`GradScaler.state_dict` 有 `hasattr` 守卫；`requirements.txt` 去掉 torch 2.2.2/torchvision 0.17.2 pin，服务器保留 1.12.1+cu113/0.13.1；`timm` 只用于 RandomErasing。未在 1.12.1 实机验证。
- 下一步：1ep smoke（λ=0 严格 V2；λ=0.1 finite/DDP/clip/head 初始化正常）；通过后再 120ep。
