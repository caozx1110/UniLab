# Resume And Checkpoints

Checkpoint selection is controlled by algorithm-level fields. Use
`algo.load_run`, not `training.load_run`.

## Resume Training

PPO/APPO resume from `algo.load_run` directly. Use a run id or `-1` for the
latest run in the relevant log directory:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco \
  algo.load_run=-1 \
  training.no_play=true
```

Off-policy algorithms (SAC/FlashSAC/WarpSAC) keep resume opt-in so that the
`algo.load_run=-1` default can never turn a fresh launch into an accidental
resume. Set `algo.resume=true` and select the checkpoint with `algo.load_run`
(run id or `-1` for the latest run) plus optional `algo.checkpoint`
(iteration or filename; `-1` selects the latest checkpoint):

```bash
uv run train --algo flashsac --task g1_motion_tracking --sim mjwarp \
  --profile <owner-profile> \
  algo.resume=true \
  algo.load_run=2026-10-05_14-28-04_mjwarp \
  training.no_play=true
```

By default, the off-policy checkpoint restores the full learner state (networks,
optimizers, schedulers, normalizers, and update count), and training
continues with absolute iteration numbering (`model_32000.pt` resumes at
iteration 32001, so `algo.max_iterations` stays the total budget, not an
increment). The replay buffer is not checkpointed: a resumed run re-warms
from an empty buffer through the usual train-start threshold, and RNG state
is not restored.

### Complete runtime continuation (source profile)

The explicit `++training.save_continuation_state=true` option adds complete
single-rank tensor-runtime snapshots. The learner, task terms and simulation
backend must implement `export_continuation_state()` and
`import_continuation_state(state)`; unsupported owners fail closed. FlashSAC
whole-cycle/DP CUDA graphs and Inductor CUDA graphs are not supported by this
continuation path. Ordinary policy checkpoints remain available for evaluation.

At each save point, the runner holds the next published inference request,
fences collector metrics and pauses replay. It preserves the physical state,
manager/observation/action histories, exploration state, valid replay ring,
already-prefetched samples, reward-normalization cursor and process RNGs. A
`model_N.continuation.json` manifest binds the native model and three component
files under `continuation_N/`. All files are synced before the manifest commits;
a partial directory without a complete manifest is not a recovery point.

Resume with the same option and the original checkpoint selector into a **new**
output directory. After cold construction/compilation, the runtime restores all
state, creates new empty IPC objects at the saved tick, and republishes the saved
observation without resetting or taking an extra environment step. It records
the restored boundary in the new output before the first update, then repeats
the checkpoint lifecycle hook so an interrupted evaluation is not skipped.
Task owners must bind immutable data/config/source identities and account for
snapshot I/O, startup and repeated evaluation costs. Large replay snapshots can
require substantial host memory and disk space. Loading is only for trusted
runtime-produced files, with SHA validation before deserialization.

## Checkpoint Format Support Boundary

PPO/APPO resume loads UniLab-patched rsl-rl checkpoints. A supported
checkpoint carries `unilab_logger_state` (the logger `tot_time` /
`tot_timesteps` counters, required) and, when the task environment publishes
versioned training state, a matching `unilab_env_training_state`. UniLab has
written both keys since the resume-state patch (issue #441, shipped since
v1.0.0), and new checkpoints keep writing them unchanged.

Starting with the first UniLab release after 1.3.5 that includes issue #2416,
loading fails closed:

- a checkpoint without a valid `unilab_logger_state` (a pre-patch or vanilla
  rsl-rl checkpoint) is rejected with an error naming the minimum supported
  format, instead of silently restarting training progress from 0;
- when the environment publishes training state, a checkpoint without a valid
  versioned `unilab_env_training_state` is rejected as well;
- an environment without exportable training state writes no placeholder key
  and requires none.

There is no compatibility window, offline migration tool, or dual-write path,
and the fail-closed behavior is not backported to a 1.3.5 patch release. To
recover an unsupported pre-patch checkpoint, retrain with a supported UniLab
version or resume from a checkpoint saved by one.

Checkpoint files are immutable load inputs: the loader never repairs,
rewrites, or migrates them.

## Replay A Checkpoint

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
uv run eval --algo sac --task g1_walk_flat --sim mujoco --load-run -1
```

`uv run eval` maps `--load-run` to the underlying checkpoint selector and sets
playback mode:

```bash
uv run eval --algo ppo --task go2_joystick_flat --sim mujoco --load-run -1
```

Some script paths accept a checkpoint path through `algo.load_run`; the unified
CLI validates `--load-run` as a run id and does not accept path separators.

## Seeds

Training seed resolution is implemented in `src/unilab/training/seed.py`.
Only the algorithm-level `algo.seed` is honored; the former `training.seed`
fallback was removed, and a config that sets `training.seed` without
`algo.seed` fails closed. The helper records seed metadata when experiment
tracking is active.
