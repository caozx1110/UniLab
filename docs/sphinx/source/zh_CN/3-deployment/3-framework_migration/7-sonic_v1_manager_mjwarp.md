# SONIC v1 Manager-Based + MJWarp 复现状态

本文记录当前 `feat/sonic-v1-manager-mjwarp` 分支的 SONIC v1.0 迁移边界、
实测吞吐和长训启动方式。上游参考代码位于
`/data/hdd/home/caozx/ws/GR00T-WholeBodyControl`，UniLab 基线为
`upstream/dev/issue-1042-manager-based-api`。

```{important}
当前结果证明的是 Manager-Based 环境、MJWarp、8-rank DDP、SONIC v1 网络和
完整 PPO update 可以在本机真实数据上运行；由于地形、事件随机化和部分 reward
仍与上游 Isaac Sim 不同，不能把当前 checkpoint 宣称为上游科学结果的等价复现。
```

## 原版与当前设置

| 项目 | 上游 SONIC v1 release recipe | 当前 UniLab 设置 |
| --- | --- | --- |
| 运行时 | Isaac Lab / Isaac Sim，`trimesh` terrain | UniLab Manager-Based API + MJWarp，当前为 flat scene |
| 机器人/控制 | G1 29 DoF，physics 200 Hz，control 50 Hz | 相同 DoF 和 `sim_dt=0.005`、`ctrl_dt=0.02` |
| rollout | 4096 env/rank，24 steps/env | 相同；8 卡目标为 32768 global env、786432 transitions/iteration |
| policy ABI | UniversalToken release model | actor 930、critic 1645、tokenizer 1761、action 29 |
| PPO | 5 epochs × 4 minibatches，100000 iterations | 相同；effective optimizer LR 为 2e-5，adaptive range 1e-5--2e-4 |
| 数据 | BONES-SEED robot/SMPL paired corpus | 本地 manifest：131418 clips、48042726 frames、50 Hz |
| 分布式 | 一进程一卡 | `torchrun`/NCCL，一进程一卡，rank-local GPU 和 CPU 绑定 |
| 加速 | 上游 Isaac Sim GPU pipeline | MJWarp GPU physics + Numba fixed-layout observation assembly |

上游 YAML 同时声明 `critic_learning_rate=1e-3`，但其 TRL trainer 实际只读取
`actor_learning_rate=2e-5` 创建单一 optimizer；当前 `SonicPPO` 保持这一有效语义。

## 已完成的工程链路

- Manager-Based task/backend owner：`task=sonic_g1_tracking/mjwarp`，并保留
  `.../mujoco` 对照 owner。
- release named model、FSQ tokenizer、五项 auxiliary loss、critic RMS、PPO
  clipped value loss、adaptive KL 和 checkpoint/resume。
- actor/critic observation 顺序、pelvis anchor、noise 范围和 930/1645/1761
  维度已按上游配置核对。
- 自适应 anchor/body height、完整 anchor quaternion、双踝 XYZ 终止和 clip-end
  timeout 已接入；running reference root height 使用上游 α=0.1 EMA。
- 完整 manifest 的 lazy/cache 读取、Numba tokenizer assembler、DDP 梯度同步和
  rank-local CPU 资源配置已验证。

最近原子提交（均已推送到 fork）包括：

```text
34a677e9  observation ABI / pelvis anchor
cd569deb  v1 tracking terminations
5ee13613  running root-height EMA
1b4b4c3d  effective release learning-rate provenance
```

## 实测吞吐（4090）

指标均来自 `config_release`、4096 env/rank、24-step rollout、完整 manifest；第
一轮包含 Warp/kernel 初始化，下面列稳态第二轮。

| 后端/拓扑 | collection | env step | learning | iteration | global iteration throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| MJWarp，1 卡 | 5.97 s | 5.53 s | 1.08 s | 7.05 s | 13946 env-step/s |
| MuJoCo，1 卡 | 6.66 s | 6.23 s | 1.08 s | 7.74 s | 12695 env-step/s |
| MJWarp，7 卡 | 7.77 s | 7.28 s | 3.19 s | 10.08 s | 68294 env-step/s |

因此在当前 flat task 上 MJWarp 单卡相对 MuJoCo 约少一成 iteration 时间，7 卡
并行后吞吐约为 6.8 万 global env-step/s。MJWarp 会提示并忽略
`cpu_ids`、`adaptive_chunk_size` 等 MuJoCo-only 选项，这是预期 backend isolation
行为；物理 collection 仍是 iteration 的主要耗时，单纯调整 PPO learner 不会消除
这一部分。

## 验证与当前阻塞

当前分支已通过 35 个 SONIC targeted pytest、ruff 和 `git diff --check`；单卡
MuJoCo/MJWarp、2 卡 DDP、7 卡真实规模两轮 smoke 均成功。工作树干净，远程分支为：

`https://github.com/caozx1110/UniLab/tree/feat/sonic-v1-manager-mjwarp`

GPU2 当前仍由用户 `pengxy` 的独立训练进程占用约 6.4 GiB，因此尚未启动真正的
8 卡长训；不应杀掉该进程或在其未释放时强行叠加作业。GPU2 释放后先做 2--3 轮
8 卡回归，再启动全量作业。

```bash
UV_PROJECT_ENVIRONMENT=/data/hdd/home/caozx/ws/UniLab/.venv \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run --no-sync scripts/train_sonic_manager.py \
  --config-name config_release \
  task=sonic_g1_tracking/mjwarp \
  training.devices='[0,1,2,3,4,5,6,7]' \
  algo.num_envs=4096 algo.num_steps_per_env=24 \
  algo.max_iterations=100000 \
  training.log_dir=/data/hdd/home/caozx/ws/datasets/bones-seed/sonic_release_v1_mjwarp_8x4090
```

训练过程中查看：

```bash
tail -f /data/hdd/home/caozx/ws/datasets/bones-seed/sonic_release_v1_mjwarp_8x4090/metrics.jsonl
```

## 尚未等价的科学边界

- 上游 `trimesh` terrain 尚未在 MJWarp owner 中复刻。
- `level0_4` 的 physics material、COM/default-pose、mass scale 和 interval push
  事件未完整表达。
- upper-body augmentation、head-link anti-shake 和部分 contact/reward 细节仍是
  当前 compact G1 contract 的近似或缺失。
- checkpoint 是 model/optimizer/normalizer 的 warm-start；simulator、采样器
  volatile slots 及 Python/NumPy/Torch RNG 不做 bit-exact continuation。

在固定评测集上得到 success rate、local MPJPE，并完成上述地形/事件差异审计前，
应使用“工程链路已跑通”而不是“科学结果 parity 已完成”的表述。
