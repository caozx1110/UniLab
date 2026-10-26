---
orphan: true
---

# ADR-0021: Complete off-policy continuation

- Status: Proposed (source profile; GPU acceptance required before production)
- Date: 2026-10-26
- Owners: runtime / algorithm / manager / simulation backend owners
- Supersedes: None
- Superseded by: None
- Scope: Explicit single-rank tensor-runtime continuation
- Authorization: [SONIC #30 complete-state requirement](https://github.com/caozx1110/sonic_unilab/issues/30#issuecomment-6408107517)

## Decision

完整续接以已发布但尚未回答的推理请求为跨进程边界。runtime 只调用注入的 env 和
learner continuation API，不依赖具体任务、manager 或 backend；物理状态由 UniSim
声明的可选能力管理。每个 owner 验证自己的 schema、layout 和配置，并原地址复制
可变数组。IPC/event、进程身份和编译产物重新建立，学习状态不重置。只读大数据资产
由外层冻结协议校验内容身份，不重复写入每个 checkpoint。

保存必须包含已预取 replay、未提交 ingress、绝对 tick/采样计数、奖励统计游标、
探索状态、env/manager 历史和独立 RNG。先 fence 指标并暂停 collector/replay，再复制
状态；所有组件和原生模型 fsync 后，最后原子提交 SHA 绑定的 manifest。原生模型
checkpoint 保留原评测 ABI。恢复在 cold preparation 之后、首次 action/update 之前
进行，不通过 reset/forward/重算观测来补状态；新目录保存恢复后的边界用于核验。

不支持的 backend、term、opaque state 和未经验证的 CUDA graph 路径必须拒绝。
这不是默认恢复语义的无条件替换，也不是对跨版本/配置恢复的承诺。

## Alternatives Considered

- 仅恢复 learner 并重新填充 replay：不满足完整续接合同。
- 仅保存 qpos/qvel：遗漏控制、warmstart、接触、历史、manager 和 RNG。
- 重采样下一 batch：改变已确定的采样与优化轨迹。
- 下游遍历或修改已安装模块：破坏 owner 边界和冻结运行。

## Evidence In Repo

- `rl/runtime/offpolicy/continuation.py`：持久化、SHA 与跨进程 RNG。
- `double_buffer_runner.py` / `worker.py`：held-tick 与指标 FIFO 边界。
- `rl/runtime/ipc`：新 IPC tick 初始化、ring/预取/ingress 状态。
- `managers/continuation.py` / `base/torch_env.py`：owner 状态和原地址恢复。
- `tests/rl/runtime/offpolicy/test_continuation.py`、
  `test_collector_continuation.py` 及 replay/learner/manager 对应测试。
