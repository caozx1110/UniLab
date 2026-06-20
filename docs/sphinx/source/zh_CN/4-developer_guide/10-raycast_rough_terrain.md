# Raycast Rough Terrain

UniLab 的 `G1WalkRoughRaycast` 对应 mjlab 注册任务
`Mjlab-Velocity-Rough-Unitree-G1`。mjlab 中实际使用 raycaster 的 rough velocity
任务有两个：`Mjlab-Velocity-Rough-Unitree-G1` 和
`Mjlab-Velocity-Rough-Unitree-Go1`；本任务先迁移 G1 版本。

该任务是新增的 MuJoCo-only PPO 任务，不改动 `G1WalkFlat`、`G1WalkRough` 等既有
任务。它通过 `SimBackend.create_raycaster()` 创建后端拥有的 batch raycaster，
任务层只读取稳定的 `(num_envs, num_rays)` 结果，不直接调用 MuJoCo 子类或解析 XML。

运行前需要使用包含 `BatchEnvPool.multi_ray` 的 TestPyPI 构建：

```bash
uv pip install --index-url https://test.pypi.org/simple \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  mujoco-uni==3.8.0.post1
```

训练命令：

```bash
uv run train --algo ppo --task g1_walk_rough_raycast/mujoco
```

迁移差异：

- 算法配置对齐 mjlab 的 PPO runner：MLP `[512, 256, 128]`、24 steps/env、
  PPO epoch/minibatch、KL、GAE 和 30000 iterations。
- 任务/机器人对齐 `Mjlab-Velocity-Rough-Unitree-G1`：G1 robot、pelvis raycast
  frame、rough terrain、terrain curriculum 开关和 velocity-command locomotion。
- raycast scan 对齐 mjlab `GridPatternCfg(size=(1.6, 1.0), resolution=0.1)`：
  pelvis yaw-aligned frame、187 条垂直下射线、terrain-only `geom_groups=[0]`、
  `cutoff=5.0` 和 `scale=1/5.0`。
- task 使用 Hydra `env.raycast_scan` 配置 frame、pattern、geom group、cutoff 和
  scale；terrain/XML 仍由 scene materialization 冷路径处理。
- 该任务只注册 MuJoCo 后端。Motrix 没有等价 batch raycast API 时应保持未实现。
