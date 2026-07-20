"""GHDistillPPO 3-phase training entrypoint (Phase 10.10).

Mirrors scripts/train_rsl_rl.py's lifecycle but drives the custom GHDistillRunner, which
self-controls the GH-schema checkpoint (no optimizer state) instead of the rsl_rl
OnPolicyRunner save path. Phase selection via `phase=train|adapt|finetune`; prior-phase
inheritance via `algo.checkpoint_path=run:.../checkpoint_final.pt`.

    uv run scripts/train_gh.py phase=train
    uv run scripts/train_gh.py phase=adapt algo.checkpoint_path=<train_final.pt>
    uv run scripts/train_gh.py phase=finetune algo.checkpoint_path=<adapt_final.pt>
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unilab.training.common import ensure_registries
from unilab.training.seed import apply_configured_training_seed


@hydra.main(version_base="1.3", config_path="../conf/gh_distill", config_name="config")
def main(cfg: DictConfig) -> None:
    from unilab.algos.gh_distill_ppo.runner import GHDistillRunner
    from unilab.training.run import get_log_root
    from unilab.utils.device import get_default_device

    ensure_registries()
    apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    device = str(cfg.training.device or get_default_device())

    log_root = get_log_root(str(hydra.utils.get_original_cwd()), cfg)
    log_dir = log_root / str(cfg.training.task_name) / f"{cfg.algo.phase}"

    runner = GHDistillRunner(cfg, device=device, log_dir=log_dir)
    runner.start()
    runner.learn()
    runner.evaluate()
    runner.close()


if __name__ == "__main__":
    main()
