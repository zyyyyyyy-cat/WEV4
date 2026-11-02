# WEV4 / GAC：V2 复位 + Gallery-Conditioned Auxiliary Classifier

代码基线：`WEV2/WHU`（HPD）。保留 `P_id`/`P_scene`、warm start、count EMA、DDP、诊断；新增 3 个 gallery 条件辅助头。

## 损失

```
L_total = CE + WRT + 0.01*ProCA + 1.0*GPD + aux_w * L_aux
L_aux   = mean_{query n, head m} CE( cos(f_n, W_m[c]) / 0.03 )
```

- `W ∈ [M,C,D]`，无 bias；warm start 后从 `P_scene[:,m]` 初始化；每个头作用在所有 query 模态上。
- `aux_w`: epoch 1 `0.05` → epoch 5 `0.10`，之后 constant。
- 训练-only；推理仍是 pre-BN CLS + L2 + 欧氏；不加载 heads/bank。

## 运行

4 卡 DDP：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/GAC_UAD.yml \
  MODEL.DIST_TRAIN True \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR ./logs/GAC_1000_DDP
```

单卡：

```bash
python train.py --config_file configs/GAC_UAD.yml \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.IMS_PER_BATCH 64 \
  SOLVER.BASE_LR 0.008
```

## 1ep smoke（必须先跑）

`GPD_DIAG_PERIOD=1` 让 HPD/GAC 诊断在第一个 iteration 就执行，避免诊断日志/诊断路径的问题拖到 120ep 才发现。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/GAC_UAD.yml \
  MODEL.DIST_TRAIN True \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  SOLVER.MAX_EPOCHS 1 \
  SOLVER.CHECKPOINT_PERIOD 1 \
  MODEL.GPD_DIAG_PERIOD 1 \
  OUTPUT_DIR ./logs/GAC_smoke
```

检查：`GAC diag` 正常打印；aux loss finite；DDP/clip 正常；`checkpoint_latest.pth` 可保存。

## 中断续训

每个 `CHECKPOINT_PERIOD`（配置为 5）以及最后一个 epoch 会保存：

- `checkpoint_latest.pth`：model + optimizer + scheduler + scaler + bank + RNG + epoch；
- `transformer_{epoch}.pth`：model-only，供 test.py 使用。

从最近一次完整状态继续：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/GAC_UAD.yml \
  --resume ./logs/GAC_1000_DDP/checkpoint_latest.pth \
  MODEL.DIST_TRAIN True \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR ./logs/GAC_1000_DDP
```

也可以 `SOLVER.RESUME ./logs/GAC_1000_DDP/checkpoint_latest.pth`，不用 `--resume`。

- 续训从 checkpoint 的 `epoch+1` 开始；DataLoader 从 epoch 边界恢复，不保证 mid-epoch 精确。
- `checkpoint_latest.pth` 每 5 epoch 写一次（`CHECKPOINT_PERIOD`），崩溃最多丢 4 个 epoch；要更细可设 `SOLVER.CHECKPOINT_PERIOD 1`。
- 完整 checkpoint 会恢复 torch/cuda/numpy/python RNG，sampler 序列从 checkpoint 继续；bank type 必须匹配，否则报错。
- 若传入 `transformer_{epoch}.pth`（model-only），只恢复模型，optimizer/scheduler/bank 重置，从 epoch 1 开始；如果 key 不兼容会报错，不再静默从零开始；不推荐用于正式续训。
- `AUX_WEIGHT=0` 时严格 V2 路径：heads 已注册但跳过 forward/loss；不改变 V2 轨迹。

## 测试

```bash
python test.py --config_file configs/GAC_UAD.yml \
  TEST.WEIGHT ./logs/GAC_1000_DDP/transformer_120.pth \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  TEST.NECK_FEAT before TEST.FEAT_NORM yes \
  DATASETS.ROOT_DIR /project/code/data
```

GD 用同一 `WHU-MARS` + `DATASETS.FILTER_GD True`，不是单独的 `WHU-MARS_1000GD` 目录。

## 2337

2337 是**独立数据集、独立训练**：`configs/GAC_2337.yml` 已设 `DATASET_DIR: WHU-MARS_2337`、`OUTPUT_DIR: ./logs/GAC_2337_DDP`。不要加载 1000 的 checkpoint，也不要对 2337 加 `DATASETS.FILTER_GD`。

**`GAC_UAD.yml` 本身可以复用**；`GAC_2337.yml` 只是把 2337 的 dataset/output 固定下来，防止训练/续训/测试时漏加 override。复用 `GAC_UAD.yml` 的完整训练命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/GAC_UAD.yml \
  MODEL.DIST_TRAIN True \
  DATASETS.DATASET_DIR WHU-MARS_2337 \
  DATASETS.ROOT_DIR /project/code/data \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  SOLVER.MAX_EPOCHS 120 \
  OUTPUT_DIR ./logs/GAC_2337_DDP
```

先检查 2337 是否有缺模态 identity（当前 PKM sampler 假定每个 pid 三模态齐全；`cur=0` 会 `divmod(...,0)`）：

```bash
python - <<'PY'
from datasets.whu_mars import WHU_MARS
from collections import defaultdict
d = WHU_MARS(root='/project/code/data', dataset_dir='WHU-MARS_2337',
             modalities=['RGB', 'IR', 'Thermal'], verbose=True)
cnt = defaultdict(lambda: defaultdict(int))
for m, rows in d.train.items():
    for _, pid, _, _ in rows:
        cnt[pid][m] += 1
bad = [(pid, [m for m in d.modality_ls if cnt[pid][m] == 0])
       for pid in cnt if any(cnt[pid][m] == 0 for m in d.modality_ls)]
print('train pids', len(cnt), 'missing-modality pids', len(bad))
print(bad[:20])
PY
```

若 `missing-modality pids > 0`，先停，当前 sampler 会失败；需要加 skip/fallback 再跑。

训练（从 ImageNet 从零训练 2337）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/GAC_2337.yml \
  MODEL.DIST_TRAIN True \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR ./logs/GAC_2337_DDP
```

测试：

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/GAC_2337.yml \
  MODEL.DIST_TRAIN False \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  TEST.WEIGHT ./logs/GAC_2337_DDP/transformer_120.pth \
  TEST.NECK_FEAT before \
  TEST.FEAT_NORM yes \
  OUTPUT_DIR ./logs/GAC_2337_test
```

## 服务器兼容性（PyTorch 1.12.1 + CUDA 11.3）

- 代码只使用 PyTorch >=1.12 API：`torch.cuda.amp.autocast/GradScaler`、`torch.einsum`、`torch.autograd.grad`、`torch.cuda.get/set_rng_state`、`DDP`、`torch.load/save`。
- `vit_pytorch.py` 的 `torch.load(..., weights_only=False)` 有 `try/except TypeError`，1.12 会回退到无该参数的形式。
- `torch.cuda.set_rng_state` 使用 `get_rng_state` 返回的 CPU ByteTensor，不 `.to('cuda')`；1.12/2.x 都兼容。
- `requirements.txt` 已去掉 `torch==2.2.2`/`torchvision==0.17.2` 的强制 pin；服务器请保留预装 `torch==1.12.1+cu113`、`torchvision==0.13.1`。`timm` 只用于 `RandomErasing`，`timm>=0.9,<1.1` 即可（服务器已有版本可直接用）。
- 未在 1.12.1 实机验证；本地是 torch 2.8.0+cpu。建议 1ep smoke 先在服务器确认 DDP/NCCL/AMP。

## 关键约束

- `GPD_TYPE='hierarchical'`、`GPD_WARMSTART=True`、`GPD_SUB_CENTERS==M`、`TEST.NECK_FEAT='before'`；否则报错。
- `aux_heads` 在 `build_transformer.__init__` 零初始化注册；warm start + bank broadcast 后所有 rank 从 `P_scene` 填充。
- `return_aux=False` 默认，`diagnose_gpd_gradients` 的 3 元组不变；mode≥1 推理路径不变。
- `aux_heads` 在 `load_param` 中跳过；test 不受影响。
