# SONIC v1 实验设置对比与进度报告

本报告比较上游公开的 SONIC v1 release recipe 与当前 UniLab 全量训练实验，并记录
截至 **2026-08-27 13:28 CST** 的静态进度快照。操作方法与稳定 contract 见
{doc}`7-sonic_unilab`；本页只负责实验口径、差异和阶段性证据。

```{important}
上游仓库没有公开 release checkpoint 的历史 GPU 型号、实际节点数和完整 launch
记录。因此，“原版设置”在本报告中严格指固定提交上的**公开 release recipe**，
不是对未公开历史训练作业的推断。
```

## 结论摘要

| 判定项 | 当前状态 | 证据边界 |
| --- | --- | --- |
| 端到端执行复现 | 已完成 | 真实 BONES-SEED G1/SMPL paired corpus、8-rank rollout、完整 PPO update 与全局 adaptive sampler 已连续运行，checkpoint 持续产出 |
| 全量从零训练 | 进行中 | `3388 / 100000` iterations，完成 `3.388%`；最新周期 checkpoint 为 iteration 3000，恢复推进审计尚待完成 |
| 运行健康 | 当前无执行故障 | 23 个 scalar 序列的全部 3388 个已记录点均为 finite；未发现 OOM、NCCL error、训练 NaN 或 rank 退出 |
| 收敛与科学结果复现 | 尚未完成 | 尚未获得固定评测集上的 success rate、local MPJPE 或 100K 最终曲线；当前 reward/KL 也不能证明收敛 |
| simulator/task parity | 尚未完成 | MuJoCo plane 与上游 Isaac Sim trimesh 不等价；upper-body 拼接和部分 domain randomization 仍缺失 |

这里的“完整 PPO 执行链已跑通”不等于“SONIC v1 科学结果已复现”。当前最准确的
描述是：**公开网络、PPO 和 motion curriculum 的主要执行 contract 已在 UniLab
上接通，全量长跑稳定推进，但任务分布与最终指标尚未达到 parity gate。**

## 对比口径与证据

- 上游：`NVlabs/GR00T-WholeBodyControl` revision
  `c374bae5b9039cd0ee71377e654d11ce1bc69e1d`。
- 上游主配置：[`sonic_release.yaml`](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/c374bae5b9039cd0ee71377e654d11ce1bc69e1d/gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml)
  与 [`ppo_im_phc.yaml`](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/c374bae5b9039cd0ee71377e654d11ce1bc69e1d/gear_sonic/config/algo/ppo_im_phc.yaml)。
- 上游规模与结果描述：[`training.md`](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/c374bae5b9039cd0ee71377e654d11ce1bc69e1d/docs/source/user_guide/training.md)
  和 [`training_data.md`](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/c374bae5b9039cd0ee71377e654d11ce1bc69e1d/docs/source/user_guide/training_data.md)。
- 当前实现：UniLab revision `b0bcac5a`，配置来源为 `conf/sonic/config.yaml`、
  `conf/sonic/task/sonic_g1_tracking/mujoco.yaml` 和
  `src/unilab/algos/torch/sonic_ppo/`。
- 当前运行证据：run 目录中的 `sonic_preflight.json`、TensorBoard event 和
  `model_1000.pt`、`model_2000.pt`、`model_3000.pt`。进度数字是本机工件的
  时间点快照，不是配置默认值或最终 benchmark。

上游 `sonic_release.yaml` 注释将该 preset 描述为 release checkpoint 的 finetune
配置，而 Training Guide 同时给出从零训练方法。当前实验明确设置
`training.resume=null`，目标是从零执行完整的 100K PPO 过程。

## 执行架构与资源

| 项目 | 上游公开 release recipe | 当前 UniLab 实验 | 结论 |
| --- | --- | --- | --- |
| 物理仿真 | Isaac Lab / Isaac Sim；`trimesh` terrain | UniLab MuJoCo CPU simulator；`plane` | 执行后端已迁移，物理与地形不等价 |
| 机器人与控制频率 | G1 29 DoF；`sim_dt=0.005`，decimation 4，即 50 Hz control | 相同 29 DoF 与 200 Hz physics / 50 Hz control | 配置对齐 |
| 公开硬件口径 | Training Guide 建议 64+ GPU，并同时给出 8-GPU 和 64-process 示例；实际 release 训练硬件未披露 | 单机 8×RTX 4090，driver `580.178.04`，PyTorch `2.7.0+cu128`，NCCL `2.26.2` | 只能比较当前资源与官方建议，不能声称替代了某个未公开硬件 |
| 分布式拓扑 | 1 process/GPU；4096 env/process | 1 rank/GPU；NUMA mapping `[0,0,0,1,1,1,1,1]` | 8-rank NCCL all-reduce 已验证 |
| 环境数 | 4096/rank；8-rank 示例为 32768 global | 4096/rank，32768 global | 对齐 8-rank recipe |
| CPU 规划 | 未公开固定 CPU profile | 152 logical / 76 physical cores、2 NUMA；7 MuJoCo workers/rank，Torch intra/inter-op `2/1` | 为 CPU sim 显式规划 56 个 worker CPU ID |
| 当前主机用量 | 无可比公开数据 | 8 rank RSS 合计约 118 GiB；审计瞬时约 25 CPU cores；主机 RAM 503 GiB | 仍有明显内存余量 |
| 单卡显存 | 未公开 | TensorBoard 最近 100 iter 的 Torch peak 均值约 8.17 GiB；`nvidia-smi` 常驻约 9.6–9.8 GiB | 24 GiB 4090 无需启用 microbatch 变体 |

GPU 利用率随 MuJoCo rollout 和 learner 阶段交替，不应把瞬时高值描述为持续满载。
本报告以 end-to-end iteration FPS、rollout FPS、learner FPS 和 rank skew 评价执行
效率。

## 数据与 motion curriculum

| 项目 | 上游公开 release recipe | 当前 UniLab 实验 | 状态 |
| --- | --- | --- | --- |
| 数据来源 | BONES-SEED robot filtered PKL 与 paired SMPL；官方文档称原始集 142,220 clips、约 288 小时，过滤后约 130K | 本地严格配对后 131,418 clips、48,042,726 frames、50 FPS | 当前是该公开数据子集的实测 materialization，不把本地数量反推为上游精确数量 |
| 存储/读取 | PKL motion library，active loading | versioned manifest + trusted receipt + 14-field shared global mmap | I/O 实现不同；manifest 显式固定当前 UniLab run 的 motion-index contract，不推断与上游 loader 的 index 顺序完全相同 |
| mmap payload | 不适用 | 77,637,047,008 bytes；source manifest SHA256 `4f14a56f21c6ca9cfae39edede14fb9b008e14cfb1e8d1c4d30b1a3a8b5d7656` | 全量 checksum/shape cold-path gate 已完成 |
| rank 数据视图 | 每 rank 从全量分布装载 active slots | `motion_shard_clips=false`，8 rank 映射同一完整只读 mmap；`motion_hot_fields=[]` | 避免每 rank 独立 materialize 一份 payload，同时保留全局 bin 坐标；各进程仍有 mmap 映射并依赖 OS page cache |
| active motion pool | `min(num_envs,1024)`，本配置为 1024/rank | 1024/rank | 对齐 |
| active pool 重采样 | 每 250 completed iterations | 每 250 iterations | 对齐 |
| adaptive bins | clip-local 50-frame bins，uniform mix 0.1 | 1,023,554 个 50-frame bins，uniform mix 0.1 | 对齐主要采样规则 |
| sampler 跨卡同步 | 每 200 iterations 平均 counters | 每 200 iterations 同步全局 counters | cadence 对齐；layout hash 为 `e5b4ccd6f84840189938a9c7c551faae29754e89f9777619c5564f778856a445` |
| freeze-frame | 启用，实际 loader 概率 0.1 | 启用，概率 0.1 | 对齐 |
| upper-body 拼接 | 启用，概率 0.5，作用于配置的 motion-key prefixes，并要求 non-nav 数据集 | 未实现；本地 manifest 也没有可证明等价的 prefix/non-nav 标注 | 未对齐 |
| temporal context | actor/critic proprioception 与 action history 均为 10；G1 future `10×0.1s`，SMPL future `10×0.02s` | 同一 release observation ABI | 对齐 |

当前正式实验选择 shared global mmap，是为了在 503 GiB 主存内让八个 rank 使用相同
的 full-corpus adaptive bin 坐标，并把全局 counters 写入 rank-zero checkpoint。
默认 rank-shard resident-hot 路径仍是低内存备选，但其 rank-local curriculum 不应
作为本次 parity run 的数据布局。

## 网络与 PPO

| 项目 | 上游公开 release recipe | 当前 UniLab 实验 | 状态 |
| --- | --- | --- | --- |
| policy 输入族 | G1、teleop、SMPL 三 encoder，均匀采样 | 同名三 encoder，均匀采样 | 对齐 |
| encoder widths | 三者均 `[2048,1024,512,512]` | 相同 | 对齐 |
| actor/critic ABI | release named UniversalToken contract | actor `930`、critic `1645`、tokenizer `1761`、action `29`，由 `sonic_release_named_universal_token.v1` 强制校验 | shape 与字段 contract 对齐 |
| tokenization | FSQ，最多 2 tokens，32 levels | 2 tokens，32 levels | 对齐 |
| decoder/critic widths | dynamic decoder 与 critic `[2048,2048,1024,1024,512,512]`；kinematic decoder `[2048,1024,512,512]` | 相同 | 对齐 |
| rollout | 24 steps/env | 24 steps/env | 对齐 |
| PPO update | 5 epochs × 4 minibatches | 5 epochs × 4 minibatches | 20 logical optimizer steps/iteration，未启用梯度累积变体 |
| 每 iteration 样本 | 8-rank 示例为 `32768×24=786432` transitions | 786,432 transitions | 对齐 |
| 核心超参数 | gamma 0.99、lambda 0.95、clip 0.2、entropy 0.01、desired KL 0.01、max grad norm 0.1 | 相同 | 对齐 |
| learning rate | effective optimizer 初始 `2e-5`；adaptive range `[1e-5,2e-4]` | 相同初始值、范围与 KL 调节规则 | 对齐；上游 YAML 的 critic `1e-3` 声明未形成独立 optimizer LR |
| auxiliary losses | 5 个 release MSE auxiliary/cycle losses | 同名 5 项与相同系数 | 对齐 |
| 训练长度 | 100,000 iterations | 100,000 iterations | 对齐 |
| checkpoint cadence | regular checkpoint 每 2000 iterations，rolling `last.pt` 每 50 | periodic checkpoint 每 1000 iterations；正常结束写 `last.pt` | 有意不同；1000 是 200-step sampler sync cadence 的整数倍 |

## Task、仿真与恢复语义

| 项目 | 上游公开 release recipe | 当前 UniLab 实验 | 状态 |
| --- | --- | --- | --- |
| reward | tracking、VR 5-point、action rate、joint limit、contact、anti-shake 和 feet acceleration release weights | `conf/sonic/task/sonic_g1_tracking/mujoco.yaml` 中的主要 terms/weights 对齐 | 主要配置对齐，跨 simulator 数值不能直接视为相同分布 |
| termination | adaptive root height/orientation、anchor/EE/feet thresholds 和 clip timeout | 主要阈值、EMA 与 clip timeout/value correction 已接通 | 主要 contract 对齐 |
| domain randomization | startup joint defaults、mass、COM、static/dynamic friction、restitution；interval push | 当前只映射 joint default、named body mass 和 dynamic friction | 缺 COM、push、static friction、restitution、bucket/startup timing 语义 |
| terrain | `trimesh` | `plane` | 未对齐 |
| checkpoint 内容 | model、optimizer、trainer/env adaptive state | model、optimizer、normalizer、iteration；global-mmap sampler counters | 当前全量 sampler state 已通过 CPU load 与 layout/finite 审计；尚未做 restart-and-advance |
| exact resume | 上游也依赖其 simulator/loader 状态语义 | simulator state、Python/NumPy/Torch RNG 与 volatile active slots 不做 bit-exact restore | 只能称 contract-level restore，不能称逐 bit 科学续跑 |
| 公开评测口径 | Training Guide 声称 well-converged policy 在 100K 后达到 success rate `>0.98`、local MPJPE `<29 mm` | 尚未执行 fixed evaluation | 这是上游文档给出的目标，不是当前结果，也不是仓库内复跑证据 |

## 当前长跑进度

本次 run ID 为
`sonic_release_v1_global_mmap_full_train_from_scratch_8x4090`，tmux session 为
`sonic-v1-full`。以下是 2026-08-27 13:28 CST 的冻结快照；训练在该时间点后继续
推进：

| 指标 | 快照值 |
| --- | --- |
| iteration | `3388 / 100000`（`3.388%`） |
| 已采 transitions | 2,664,431,616 / 78,643,200,000 |
| 已执行 optimizer updates | 67,760 / 2,000,000 |
| 已运行时间 | 约 17 小时 14 分钟 |
| 稳态速度 | 约 18.18 秒/iteration，约 198 iterations/小时 |
| 最近 100 iter rollout FPS | 55,465 均值 |
| 最近 100 iter learner FPS | 127,765 均值 |
| 最近 100 iter end-to-end iteration FPS | 38,655 均值 |
| collect / train rank skew | 1.160 / 1.460 均值 |
| checkpoints | `model_1000.pt`、`model_2000.pt`、`model_3000.pt` |
| 最新周期 checkpoint | iteration 3000；`algorithm.update_count=60000`，每 iter 20 updates；恢复推进审计尚待完成 |
| 粗略剩余时间 | 若吞吐不变约 20.4 天；中断、评测和后续吞吐变化均未计入 |

三个 checkpoint 均已完成 CPU load 与递归 finite audit。`model_3000.pt` 保存了
131,418 clips / 1,023,554 bins 的 shared-global-mmap sampler 状态，layout hash 与
当前数据一致，episode/failure counters 为 finite 且持续累积。日志检索没有发现
Traceback、OOM、NCCL error、segfault 或 rank 退出。

### 训练信号与风险

| 信号 | 当前观察 | 解读边界 |
| --- | --- | --- |
| reward/mean | 首 100 iter 均值约 0.0813；最近 100 iter 均值约 0.0612 | reward 随每 250 iter 重采的 active pool/curriculum 改变，不能据此直接断言性能退化；但也没有证据声称正在收敛 |
| approximate KL | 最近 100 iter 均值约 0.0219，desired KL 为 0.01 | 已处于目标的约两倍，是需要继续监控并用评测交叉验证的信号 |
| learning rate | 最近窗口固定在下限 `1e-5` | adaptive scheduler 已没有继续降 LR 的空间；后续应结合 KL 和 fixed evaluation 判断，而不是在当前 canonical run 中途改参 |
| action noise std | 最近 100 iter 均值约 0.2384 | 仍为 finite，需与 checkpoint 策略质量一起评估 |
| 数值健康 | 23 个 scalar 序列的全部 3388 个已记录点均无 NaN/Inf | 只证明数值与执行健康，不证明最终任务质量 |

## 完成项与剩余 gate

已完成：

- 8×4090 driver、NCCL、NUMA 与 rank-local CPU 资源 gate；
- BONES-SEED G1/SMPL 全量配对、manifest/checksum/shape 审计和 14-field global mmap；
- release named model、PPO update、distributed normalization、global adaptive sampler、
  freeze-frame 和 checkpoint 执行链；
- 100-clip strict resume、全量 corpus smoke，以及全量从零训练到 iteration 3000 以上；
- 三个周期 checkpoint 的 load、shape、sampler state 和 finite audit。

进行中：

- 100,000 iteration 全量长跑；
- 每个新 checkpoint 的完整性检查和训练信号监控。

尚未完成：

- 对 `model_1000/2000/3000` 使用同一 fixed evaluation corpus，得到可比较的
  success rate、local MPJPE 和 episode-level failure 分解；
- 从全量 `model_3000.pt` 启动独立的 restart-and-advance 审计，确认恢复后 sampler
  counters 连续推进；
- 100K 最终曲线与上游 Training Guide 所述目标的复核；
- trimesh terrain、upper-body 拼接以及缺失的 domain randomization；
- 跨 Isaac Sim/MuJoCo 的动力学误差审计和 sim2sim evaluation；
- simulator 与全部 RNG 状态的 bit-exact resume。

下一里程碑应先在不停止训练的情况下评测现有三个 checkpoint。若 fixed evaluation
也显示停滞，再从当前 canonical run 分叉独立调参实验；不要用中途修改 KL/LR 的
结果覆盖这条从零基线。科学 parity 的完成条件是：完整训练结束、固定评测指标可
复核，并且所有仍保留的 task/simulator 差异均被明确量化或关闭。

## 查看本机运行

```bash
tmux attach -t sonic-v1-full
```

TensorBoard 工件位于本机 run 目录的 `tb/`：

```bash
RUN_DIR=/data/hdd/home/caozx/ws/datasets/bones-seed/sonic_release_v1_global_mmap_full_train_from_scratch_8x4090
uv run tensorboard --logdir "$RUN_DIR/tb" --port 6006
```
