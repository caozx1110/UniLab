# 续训与检查点

检查点选择由算法层面的字段控制。请使用 `algo.load_run`，而不是
`training.load_run`。

## 续训

PPO/APPO 直接通过 `algo.load_run` 续训。使用 run id，或用 `-1` 表示相关日志目录中最新的一次运行：

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.load_run=-1 \
  training.no_play=true
```

off-policy 算法（SAC/FlashSAC/WarpSAC）的续训是显式开启的，以避免
`algo.load_run=-1` 的默认值把全新启动意外变成续训。设置 `algo.resume=true`，并用
`algo.load_run`（run id 或 `-1` 表示最新 run）与可选的 `algo.checkpoint`
（迭代号或文件名；`-1` 表示最新 checkpoint）选择检查点：

```bash
uv run train --algo flashsac --task g1_motion_tracking --sim mjwarp \
  --profile <owner-profile> \
  algo.resume=true \
  algo.load_run=2026-10-05_14-28-04_mjwarp \
  training.no_play=true
```

默认 off-policy checkpoint 恢复完整的 learner 状态（网络、优化器、调度器、normalizer
与 update count），训练以绝对迭代编号继续（`model_32000.pt` 从第 32001 次迭代继续，
因此 `algo.max_iterations` 仍是总预算而不是增量）。replay buffer 不落盘：续训的 run
会从空 buffer 重新经过 train-start 阈值热身，且 RNG 状态不恢复。

### 完整运行时续接（源码 profile）

显式设置 `++training.save_continuation_state=true`，为单 rank tensor runtime
保存完整状态。learner、任务 term 和仿真 backend 必须实现
`export_continuation_state()` / `import_continuation_state(state)`；不支持的 owner
明确拒绝，不隐式降级。该路径暂不支持 FlashSAC whole-cycle/DP CUDA graph 或
Inductor CUDA graph。原生模型 checkpoint 保持评测兼容。

保存时，runner 保留下一条已发布推理请求，排空该边界之前的 collector 指标并暂停
replay，保存物理状态、manager/观测/动作历史、探索状态、有效 replay ring、已经
预取的样本、奖励归一化游标及各进程 RNG。`model_N.continuation.json` 绑定原生模型
和 `continuation_N/` 内三个组件文件；全部文件同步落盘后才提交 manifest。
没有完整 manifest 的半成品目录不能用于恢复。

恢复保留同一选项及原 checkpoint 选择器，必须使用**新的输出目录**。构建/编译后
再恢复状态，新建空 IPC 对象并从保存的 tick 接续，只重新发布保存的观测，不 reset、
不多走环境步。首次更新前在新目录再次保存恢复边界，并重做 checkpoint lifecycle
hook，避免跳过被抢占中断的评测。任务 owner 仍须绑定不可变数据、配置和源码身份，
计入保存 I/O、初始化与重复评测成本，并核查大型 replay 对主机内存和磁盘的需求。
只加载本运行时产生的可信文件，先校验 SHA，再反序列化。

## 检查点格式支持边界

PPO/APPO 续训加载的是 UniLab patch 过的 rsl-rl checkpoint。受支持的 checkpoint 必须
携带 `unilab_logger_state`（logger 的 `tot_time` / `tot_timesteps` 计数器，必需），
并且当任务环境发布版本化 training state 时，还需携带对应的
`unilab_env_training_state`。自 resume-state patch（issue #441，v1.0.0 起发布）以来
UniLab 一直写入这两个键，新保存的 checkpoint 保持不变。

从包含 issue #2416 的 1.3.5 之后的第一个 UniLab 版本起，加载改为 fail closed：

- 缺少有效 `unilab_logger_state` 的 checkpoint（pre-patch 或原生 rsl-rl checkpoint）
  会被拒绝加载，错误信息说明最低支持格式，不再静默地把训练进度从 0 恢复；
- 当环境发布 training state 时，缺少有效版本化 `unilab_env_training_state` 的
  checkpoint 同样会被拒绝；
- 没有可导出 training state 的环境不写入占位键，也不要求该键。

不提供兼容窗口、离线迁移工具或 dual-write 路径，该 fail-closed 行为也不会回 port 到
1.3.5 patch release。对于不受支持的 pre-patch checkpoint，请用受支持的 UniLab 版本
重训，或从该版本保存的 checkpoint 续训。

checkpoint 文件是只读加载输入：加载器不会修复、重写或迁移它们。

## 回放检查点

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1
```

`uv run eval` 会将 `--load-run` 映射到底层的检查点选择器，并设置回放模式：

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

某些脚本路径接受通过 `algo.load_run` 传入的检查点路径；统一 CLI 会将 `--load-run`
校验为一个 run id，且不接受路径分隔符。

## 随机种子

训练种子的解析在 `src/unilab/training/seed.py` 中实现。仅认可算法级
`algo.seed`；旧 `training.seed` fallback 已删除，配置了 `training.seed`
但缺少 `algo.seed` 时会 fail closed。在实验跟踪启用时，该辅助逻辑会记录
种子元数据。
