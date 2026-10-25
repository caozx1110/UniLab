"""Owner-config resolution stays outside the RL runtime."""

import pytest
from omegaconf import OmegaConf

from unilab.training.lifecycle import resolve_offpolicy_lifecycle


def factory(*, value):
    def hook(context):
        context.append(value)

    return hook


def test_factory_and_explicit_milestones():
    cfg = OmegaConf.create(
        {
            "lifecycle": {
                "factory": f"{__name__}:factory",
                "kwargs": {"value": 9},
                "checkpoint_iterations": [2, 7],
            }
        }
    )
    hook, iterations = resolve_offpolicy_lifecycle(cfg)
    actual = []
    hook(actual)
    assert actual == [9] and iterations == (2, 7)


def test_no_lifecycle_and_save_only_config():
    assert resolve_offpolicy_lifecycle(OmegaConf.create({})) == (None, ())
    assert resolve_offpolicy_lifecycle(
        OmegaConf.create({"lifecycle": {"checkpoint_iterations": [3]}})
    ) == (None, (3,))


@pytest.mark.parametrize(
    "config",
    [
        {"unknown": 1},
        {"factory": "missing-colon"},
        {"kwargs": {"x": 1}},
        {"checkpoint_iterations": [True]},
        {"checkpoint_iterations": [2, 2]},
    ],
)
def test_malformed_owner_lifecycle_rejects(config):
    with pytest.raises(ValueError):
        resolve_offpolicy_lifecycle(OmegaConf.create({"lifecycle": config}))
