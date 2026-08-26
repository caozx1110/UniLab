# SONIC 迁移到 UniLab

本页描述 `GR00T-WholeBodyControl/gear_sonic` 的 SONIC release 配置如何接入
UniLab 的多卡启动、native `SonicPPO` owner 和冷路径数据契约。训练入口默认
使用 `unilab.algos.torch.sonic_ppo:train_sonic`；`sonic.runtime_entrypoint`
只用于替换成另一个兼容 owner，不会把现有 RSL-RL PPO 当作 SONIC 替代品。

这里的“已验证”只指下文列出的真实数据、真实 8 卡执行和 checkpoint 结果；不表示
完整训练曲线、任务成功率或上游 SONIC v1 的科学结果已经等价。

## 先做 preflight

```bash
UV_CACHE_DIR=/tmp/unilab-uv-cache uv run scripts/train_sonic_unilab.py
```

默认配置对应每 rank 4096 个环境、24 步 rollout、5 个 PPO epoch 和 4 个
minibatch。`training.devices=[0,1,2,3,4,5,6,7]` 时，preflight 会记录全局
32768 环境、每轮 786432 个 transition 和每轮 20 次 logical optimizer update，
并在 run 目录写入 `sonic_preflight.json`。

## 多卡和 CPU 资源

训练模式使用 UniLab 的 `torchrun` launcher，一进程一卡；rank-local 的
`EnvCfg.cpu_ids` 会在环境 materialize 前注入 MuJoCo。目标主机 profile 为
152 logical CPU、每 rank 7 个 MuJoCo worker、Torch intra/inter-op 线程
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

目标主机已经通过真实驱动 gate：8 张 RTX 4090（driver `580.178.04`、PyTorch
`2.7.0+cu128`、NCCL `2.26.2`）均可见，8-rank NCCL all-reduce 校验通过；同
NUMA 内 GPU 为 `PIX`、跨 NUMA 为 `SYS`，所以 profile 按 GPU0--2→NUMA0、
GPU3--7→NUMA1 分配 CPU。下表是同一主机上真实 BONES-SEED G1/SMPL paired
motion 的已完成续训证据；均为 4096 env/rank、horizon=24、5×4 PPO updates，
即全局 32768 env、786432 transition/iteration。

| 数据与运行 | 已完成的 checkpoint 路径 | 最后一次 iteration 指标 |
| --- | --- | --- |
| 100 clips、global-mmap benchmark layout、freeze-frame augmentation 和 active motion pool；sampler stats 每个 PPO iteration 强制同步；`model_7.pt → model_8.pt` | `sonic_release_v1_freeze_pool_global_mmap_smoke100_resume_7_to_8_v2/model_8.pt` | rollout `54000.4` FPS；iteration `36993.7` FPS；learner `117464.2` FPS；reward `0.075`；KL `0.0072`；单卡峰值 `8.174 GiB` |
| 同一 100-clip benchmark 的 `model_8.pt → model_9.pt` strict sampler resume | `sonic_release_v1_freeze_pool_global_mmap_smoke100_resume_8_to_9/model_9.pt` | rollout `58777.6` FPS；iteration `42355.6` FPS；learner `151599.5` FPS；reward `0.074`；KL `0.0075`；单卡峰值 `8.167 GiB`。730-bin sampler layout hash 未变化，累积 counters 保持 finite。 |
| 完整 131,418 clips、rank-shard resident-hot layout、freeze-frame augmentation 和 active motion pool；`model_7.pt → model_8.pt` | `sonic_release_v1_freeze_pool_rankshard_full_resume_7_to_8/model_8.pt` | rollout `52696.0` FPS；iteration `36896.5` FPS；learner `123060.2` FPS；reward `0.084`；KL `0.0080`；单卡峰值 `8.160 GiB` |

100-clip global-mmap 结果验证了该 benchmark 的执行、sampler restore 和后续
iteration；它不是长周期收敛证据，也不能代替全量 corpus 结果。全量 rank-shard
结果验证了真实的 131,418 clips 数据和一次完整续训 iteration，但该布局中的
adaptive counters 是 rank-local，既不跨 rank 同步，也不写入 rank-zero
checkpoint；它不能作为上游全局 curriculum 的等价证据。

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
默认保留按 clip 延迟解码的低内存路径，单字段 NPY 以只读 mmap 打开。release
MuJoCo owner 则在环境初始化时把 rollout 使用的 8 个字段冷物化为 rank-local、
C-contiguous、只读数组；48,042,726 帧的完整 paired corpus 共占约 56.6 GiB，
八卡平均约 7.1 GiB/rank。未选字段仍走 lazy fallback，
`sonic.motion_cache_size=2` 只保留两个解码 clip，不复制第二份完整 shard。这样既
避免 adaptive sampler 在约 16k clips/rank 上反复解压，也让 step/reset 中的
frame gather 直接使用 `np.take`。resident/cache 只改变驻留和 I/O 行为，不改变
manifest、frame index 或 joint/body reorder contract。显式 preflight 对完整
corpus 做 checksum/shape 审计；训练启动时八个 rank 各自严格验证并冷物化将消费
的 1/8 clips，合计覆盖 corpus 一次，避免父进程和每个 rank 重复全量扫描。

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
失败。PPO 的 Gaussian KL、clipped value loss（无额外 `0.5` 因子）、FSQ
normalized code 范围与 release 算术保持一致。checkpoint 恢复模型、optimizer、
normalizer 和 iteration。上表的 global-mmap benchmark 已验证一次带 sampler
state 的严格恢复；这个实验边界不能外推到别的 manifest、数据布局或长期训练。
rank-shard 布局不会保存其 rank-local adaptive counters，故该布局的 resume 仍是
参数/optimizer/normalizer 的 warm-start，不能当作无缝、全局可复现的 resume。
环境状态及 Python/NumPy/Torch RNG 也不在 checkpoint 中。

官方 `sonic_release/last.pt` 使用 TRL pickle metadata，先转换成 UniLab
checkpoint：

```bash
UV_CACHE_DIR=/tmp/unilab-uv-cache uv run scripts/convert_sonic_release_checkpoint.py \
  --source /abs/path/sonic_release/last.pt \
  --output /abs/path/sonic_release_unilab.pt \
  --trust-source
```

已验证官方 iteration `41550`、72 个 model tensor 和 optimizer state 均可
strict load；用 `training.resume=/abs/path/sonic_release_unilab.pt` 可做
warm-start 或兼容性验证。从零复现 release 训练时不要设置 `training.resume`。

默认的 rank-shard motion corpus 让每个 rank 的 adaptive counters 独立演进，不能
将该路径的统计或 checkpoint resume 称为上游的全局同步 curriculum。100-clip
global-mmap benchmark 的 sampler restore 仅是该固定数据布局上的执行证据，不是
完整 corpus 的长期采样分布或 source-parity 证明。

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

native owner 的 PPO 生命周期、sequence minibatch、microbatch/update cadence、
named G1/teleop/SMPL encoders、FSQ、named decoders、critic 和官方 checkpoint
shape 已按 release contract 接通。任务 reward/termination 已对齐主要 release
terms，且真实 8 卡运行已覆盖 clip-end timeout 的 value correction 与当前
freeze-frame augmentation 路径。

仍未等价的边界必须保留：domain randomization 目前只映射 joint default、动态
friction 和指定 body mass；COM、interval push、static friction/restitution 和
startup timing 仍不完整。terrain 仍是 plane，而非上游 trimesh。上游的
upper-body/navigation motion 拼接尚未实现；本地 manifest 的 131,418 clips 也没有
可识别的 nav prefix，不能强行标注或拼接。随机数流及其与上游 loader、环境、
sampler 的消费顺序也未证明一致。

因此可以声称官方网络和完整 PPO 执行链兼容，并已在真实全量数据上完成一次 8 卡
续训 iteration；在完整 corpus 的长周期收敛曲线、任务成功率和上述仿真/数据/RNG
差异完成验证前，不能声称 SONIC v1 的科学结果 parity。
