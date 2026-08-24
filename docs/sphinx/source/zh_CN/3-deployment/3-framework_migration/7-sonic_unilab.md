# SONIC 迁移到 UniLab

本页描述 `GR00T-WholeBodyControl/gear_sonic` 的 SONIC release 配置如何接入
UniLab 的多卡启动、native `SonicPPO` owner 和冷路径数据契约。训练入口默认
使用 `unilab.algos.torch.sonic_ppo:train_sonic`；`sonic.runtime_entrypoint`
只用于替换成另一个兼容 owner，不会把现有 RSL-RL PPO 当作 SONIC 替代品。

## 先做 preflight

```bash
UV_CACHE_DIR=/tmp/unilab-uv-cache uv run scripts/train_sonic_unilab.py
```

默认配置对应每 rank 4096 个环境、24 步 rollout、5 个 PPO epoch 和 4 个
minibatch。`training.devices=[0,1,2,3,4,5,6,7]` 时，preflight 会记录全局
32768 环境和每轮 786432 个 transition，并在 run 目录写入
`sonic_preflight.json`。

## 多卡和 CPU 资源

训练模式使用 UniLab 的 `torchrun` launcher，一进程一卡；rank-local 的
`EnvCfg.cpu_ids` 会在环境 materialize 前注入 MuJoCo。目标主机 profile 为
152 logical CPU、每 rank 6 个 MuJoCo worker、Torch intra/inter-op 线程
2/1。目标拓扑是 rank 0--2 位于 NUMA0、rank 3--7 位于 NUMA1；上线前必须用
`nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader` 把 CUDA ordinal
映射到 `/sys/bus/pci/devices/<BDF>/numa_node`，然后在
`sonic.resources.gpu_numa_nodes` 中按 rank 显式写入。PCI BDF 排序只用于无
driver 的 preflight fallback，不能作为生产 ordinal 映射证据。生产训练应把
owner 和拓扑一起固定，例如：

```bash
UV_CACHE_DIR=/tmp/unilab-uv-cache uv run scripts/train_sonic_unilab.py \
  task=sonic_g1_tracking/mujoco \
  sonic.mode=train \
  sonic.resources.gpu_numa_nodes='[0,0,0,1,1,1,1,1]' \
  sonic.motion_manifest=/abs/path/to/manifest.json \
  sonic.require_motion_manifest=true
```

`task=sonic_g1_tracking/mujoco` 是 task/backend owner 的 Hydra 选择；
`training.sim_backend` 只记录 owner 身份，不是独立的后端切换开关。当前
SONIC 只注册 MuJoCo owner，Motrix 尚未有可 materialize 的 SONIC 适配，不能
通过 `task=.../motrix` 启动。

SONIC profile 不继承 RTX 6000D 所需的 `NCCL_P2P_DISABLE=1` /
`NCCL_SHM_DISABLE=1` workaround。4090 上应分别 benchmark 环境变量 unset、
`0/0` 和兼容 fallback `1/1`，以 rollout FPS、learner FPS、all-reduce 延迟和
rank skew 决定，不在配置里预设“最优”链路。

已在目标主机完成一次真实驱动 gate：8 张 RTX 4090（driver
`580.178.04`、PyTorch `2.7.0+cu128`、NCCL `2.26.2`）均可见，8-rank
NCCL all-reduce 校验通过；同 NUMA 内 GPU 为 `PIX`、跨 NUMA 为 `SYS`，因此
profile 按 GPU0--2→NUMA0、GPU3--7→NUMA1 分配 CPU。使用 8 个合成、已校验的
motion clip 做一轮完整 release 网络 smoke（4096 env/rank、horizon=24、
5×4 PPO updates）也成功：全局 32768 env、786432 transitions/iteration，
`perf/iteration_fps=42069.9`，`perf/rollout_fps=54359.6`，峰值 Torch 显存
`6.921 GiB`，rank skew 为 collect `1.072`、train `1.319`。GPU0--2 当时有
其他作业占用，故该吞吐是保守的混部结果；正式 benchmark 应先清空 GPU 并对
NCCL 环境变量做 A/B 扫描（本次仅设置 `NCCL_IB_DISABLE=1`，未禁用 P2P/SHM）。
该结果验证的是硬件、launcher、MuJoCo owner 和
PPO 生命周期，不代表真实 corpus 的 I/O 或训练曲线。

每个 rank 在构造模型和环境前应用 `algo.seed + rank`，保证同一 rank 可复现，
同时避免八个环境采样器得到完全相同的随机流。

native owner 按完整环境序列组织 rollout：每 rank 4096 个环境、24 步、每个
logical minibatch 1024 个环境序列。release 的
`per_device_train_batch_size=null` 会解析为 1024，因此每轮是
`5 × 4 = 20` 次 optimizer update；默认配置保持这个严格语义。如果 24 GB
显存实测 OOM，可设 `sonic.microbatch_size=128`、
`sonic.optimizer_step_per_microbatch=false`、`sonic.allow_microbatch_change=true`，
用 8 个 microbatch 做一次 logical-minibatch 梯度累积。该配置保持 20 次
optimizer update，但仍是需要单独记录和比较的资源优化变体。线程预算会在
每个 rank 真正调用 `torch.set_num_threads(2)` 和
`torch.set_num_interop_threads(1)`，并在环境构造前写入 BLAS 线程环境变量。

## motion materialization

`sonic_motion.py` 的 manifest 是冷路径契约，包含版本、关节/刚体顺序、字段
shape/dtype、clip fps 和 SHA256。规范化 NPZ clip 可以这样打包：

```bash
UV_CACHE_DIR=/tmp/unilab-uv-cache uv run scripts/materialize_sonic_motion.py \
  --source data/clip_000.npz --source data/clip_001.npz \
  --output data/sonic_store --fps 50 \
  --joint-order j0 j1 --body-order pelvis
```

release 的 robot 与 SMPL 单 clip 目录可以按唯一 basename 流式配对转换：

```bash
UV_CACHE_DIR=/tmp/unilab-uv-cache uv run scripts/materialize_sonic_motion.py \
  --robot-root /abs/path/to/robot_filtered \
  --smpl-root /abs/path/to/bones_seed_smpl \
  --output data/sonic_store --fps 50 \
  --joint-order j0 j1 --body-order pelvis \
  --fk-model /abs/path/to/g1.xml --smpl-y-up
```

converter 默认对 duplicate basename、缺失配对 fail-closed；robot 与 SMPL 按各自
fps/duration 重采样到共同 target grid，帧数仍不一致时拒绝。只有显式传入
`--allow-unmatched` 才会跳过未配对 key。每次只规范化
一个 pair，并在所有 clip 通过 checksum/shape preflight 后原子发布 store。这里的
命令是小样本用法，不代表全量 corpus 已验证。step/reset 热路径不能读取 PKL、XML
或重新解析 manifest。训练前将生成的
`manifest.json` 设置到 `sonic.motion_manifest`，并保持 checksum/shape 校验
开启。多卡默认 `sonic.motion_shard_clips=true`，每个 rank 只 materialize
round-robin 的 clip 子集，避免 8 个进程各自复制完整 corpus。rank-local store
不会再拼接子集：NPZ 按 clip 延迟解码，`sonic.motion_cache_size=2` 限制每个
rank 的 LRU 常驻 clip 数；单字段 NPY 则以只读 mmap 打开。cache 只改变驻留和
I/O 行为，不改变 manifest、frame index 或 joint/body reorder contract。实际
RSS、cache miss 和存储吞吐仍必须纳入上线 gate；若随机采样导致 NPZ 反复解压，
优先离线切换到大小均衡的 NPY shards，而不是在 `step/reset` 热路径解析源数据。

## 开始训练

```bash
UV_CACHE_DIR=/tmp/unilab-uv-cache uv run scripts/train_sonic_unilab.py \
  task=sonic_g1_tracking/mujoco \
  sonic.mode=train \
  sonic.resources.gpu_numa_nodes='[0,0,0,1,1,1,1,1]' \
  sonic.motion_manifest=/abs/path/to/manifest.json \
  sonic.require_motion_manifest=true
```

内置 owner 已包含 sequence-aware rollout、UniversalToken 的 native MLP+FSQ
实现、tokenizer reconstruction/commitment auxiliary loss、critic running
normalization、adaptive learning-rate scheduler、分布式 advantage/normalizer
同步和 checkpoint 写入。正常结束时 rank 0 总会原子写入 `last.pt`；恢复前各
rank 会校验 checkpoint SHA256 一致性，新格式缺少维度/token contract 时直接
失败。PPO 的 Gaussian KL、clipped value loss（无额外
`0.5` 因子）、FSQ normalized code 范围与 release 算术保持一致。checkpoint
目前是参数/optimizer/normalizer 的 warm-start：不会恢复环境、motion sampler
或 Python/NumPy/Torch RNG，因此不能当作无缝可复现 resume。

由于默认 motion corpus 按 clip 分片，每个 rank 的 adaptive bin 坐标也不同；
当前 curriculum 是 rank-local adaptive variant，不能把各 rank 的 bin vector
直接 all-reduce 后称为上游同步语义。精确 global adaptive 模式需要先定义稳定的
manifest-level bin 坐标，或关闭 clip sharding，再作为独立 owner contract 实现。

建议按 `512 → 1024 → 2048 → 4096 env/rank` 爬坡，再扫描 1/2/4/6/8 卡；
扫描时固定全局 transition budget，而不是固定每卡环境数。例如目标预算为
`32768 × 24` transitions/iteration 时，world size 为 1/2/4/8 分别使用
`algo.num_envs=32768/16384/8192/4096`（保持 horizon=24）。每次记录
TensorBoard 中的 `perf/rollout_fps`、`perf/learner_fps`、`perf/iteration_fps`、
rank skew、显存峰值和 `policy/learning_rate`，同时记录 CPU worker 利用率。
真实驱动和 8 卡 launcher gate 已通过；真实 SONIC motion corpus 的 checksum、
I/O/RSS、clip 覆盖率和最终训练曲线仍需单独上线 gate，不能由合成 clip smoke 或
preflight 结果代替。

## Parity 边界

native owner 的 PPO 生命周期、sequence minibatch 和 microbatch/update cadence
已按 release contract 接通，但 UniversalToken 当前是轻量 native MLP+FSQ，
不是上游多 named encoder/decoder 的 checkpoint-compatible architecture；任务
仍使用 plane 和 UniLab generic reward/termination，而 release 使用 trimesh、
专用 reward/termination/event 配置。因而当前结果是可运行的 native migration
baseline，不能宣称 checkpoint 或训练曲线 parity。named tokenizer、任务 reward/
termination/terrain、adaptive sampler 精确同步和完整 corpus 曲线需要继续按
独立 owner issue 对齐，并在真实数据上复核。
