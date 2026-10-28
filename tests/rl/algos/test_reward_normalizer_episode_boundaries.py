"""Regression cases adapted from SONIC issue #60's F1/F2 reproductions.

The opt-in is owned by configuration, so the same saved statistics can be used
for paired runs. Ordinary checkpoint loading must still support same episodes.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from unilab.rl.algos.flash_sac.learner import FlashSACLearner, RewardNormalizer
from unilab.rl.runtime.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner


def _learner(*, episode_boundaries=False, normalize_reward=True):
    return FlashSACLearner(
        obs_dim=3,
        critic_obs_dim=5,
        action_dim=2,
        device="cpu",
        gamma=0.97,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        actor_num_blocks=1,
        critic_num_blocks=1,
        use_compile=False,
        use_amp=False,
        use_whole_cycle_cuda_graph=False,
        normalize_reward=normalize_reward,
        reward_normalizer_episode_boundaries=episode_boundaries,
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_terminal_reward_is_in_episode_statistics_before_carry_reset(enabled, batched):
    normalizer = RewardNormalizer(0.97, 5.0, torch.device("cpu"), episode_boundaries=enabled)
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    if batched:
        normalizer.update_from_transitions(rewards, dones)
    else:
        carries = []
        for reward, done in zip(rewards, dones, strict=True):
            normalizer.update_from_transitions(reward, done)
            carries.append(normalizer.g_r.item())
        assert carries == pytest.approx([1.0, 0.0, 3.0] if enabled else [1.0, 2.0, 4.94])

    # Independent fixed oracle, including RMS's documented count/variance prior.
    samples = torch.tensor([1.0, 2.97, 3.0] if enabled else [1.0, 2.0, 4.94])
    count = 3 + 1e-4
    mean = samples.sum() / count
    variance = (samples.square().sum() + 1e-4) / count - mean.square()
    torch.testing.assert_close(normalizer.rms.mean, mean.reshape(1))
    torch.testing.assert_close(normalizer.rms.var, variance.reshape(1))
    assert normalizer.rms.count.item() == pytest.approx(count)
    assert normalizer.g_r.item() == pytest.approx(3.0 if enabled else 4.94)
    assert normalizer.g_r_max.item() == pytest.approx(3.0 if enabled else 4.94)


@pytest.mark.parametrize("enabled", [False, True])
def test_rebuilt_environment_resume_preserves_statistics_and_resets_only_carry(tmp_path, enabled):
    # A checkpoint from the original implementation has no new schema fields.
    old = _learner()
    old.update_reward_stats(torch.tensor([10.0, -5.0]), torch.zeros(2))
    old.update_count = 7
    checkpoint = tmp_path / "model.pt"
    torch.save(old.get_state_dict(), checkpoint)
    fresh = _learner(episode_boundaries=enabled)
    runner = SimpleNamespace(device=torch.device("cpu"), learner=fresh)
    next_iteration = DoubleBufferOffPolicyRunner._restore_resume_checkpoint(runner, str(checkpoint))
    assert next_iteration == 8
    assert fresh.update_count == 7
    actual, previous = fresh.reward_normalizer, old.reward_normalizer
    assert actual is not None and previous is not None
    torch.testing.assert_close(actual.rms.mean, previous.rms.mean, rtol=0, atol=0)
    torch.testing.assert_close(actual.rms.var, previous.rms.var, rtol=0, atol=0)
    torch.testing.assert_close(actual.rms.count, previous.rms.count, rtol=0, atol=0)
    torch.testing.assert_close(actual.g_r_max, previous.g_r_max, rtol=0, atol=0)
    torch.testing.assert_close(actual.g_r, torch.zeros(2) if enabled else previous.g_r)
    fresh.update_reward_stats(torch.ones(2), torch.zeros(2))
    torch.testing.assert_close(
        actual.g_r, torch.ones(2) if enabled else torch.tensor([10.7, -3.85])
    )
    assert actual.g_r_max.item() == pytest.approx(10.0 if enabled else 10.7)


@pytest.mark.parametrize("enabled", [False, True])
def test_same_episode_checkpoint_roundtrip_retains_carry(tmp_path, enabled):
    first = _learner(episode_boundaries=enabled)
    first.update_reward_stats(torch.tensor([1.0, 2.0]), torch.zeros(2))
    checkpoint = tmp_path / "same_episode.pt"
    torch.save(first.get_state_dict(), checkpoint)
    restored = _learner(episode_boundaries=enabled)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert restored.reward_normalizer is not None and first.reward_normalizer is not None
    torch.testing.assert_close(restored.reward_normalizer.g_r, torch.tensor([1.0, 2.0]))
    for learner in (first, restored):
        learner.update_reward_stats(torch.tensor([3.0, 4.0]), torch.tensor([0.0, 1.0]))
        learner.update_reward_stats(torch.tensor([2.0, 1.0]), torch.zeros(2))
    torch.testing.assert_close(
        restored.reward_normalizer.state_dict(),
        first.reward_normalizer.state_dict(),
        rtol=0,
        atol=0,
    )


def test_episode_reset_keeps_storage_and_supports_disabled_normalization():
    learner = _learner(episode_boundaries=True)
    learner.update_reward_stats(torch.tensor([10.0]), torch.zeros(1))
    normalizer = learner.reward_normalizer
    assert normalizer is not None
    pointers = [
        normalizer.g_r.data_ptr(),
        normalizer.g_r_max.data_ptr(),
        normalizer.rms.var.data_ptr(),
    ]
    before = copy.deepcopy(normalizer.state_dict())
    learner.reset_environment_state()
    assert normalizer.g_r.item() == 0
    assert pointers == [
        normalizer.g_r.data_ptr(),
        normalizer.g_r_max.data_ptr(),
        normalizer.rms.var.data_ptr(),
    ]
    torch.testing.assert_close(normalizer.rms.state_dict(), before["rms"], rtol=0, atol=0)
    torch.testing.assert_close(normalizer.g_r_max, before["g_r_max"], rtol=0, atol=0)
    _learner(episode_boundaries=True, normalize_reward=False).reset_environment_state()


def test_default_normalizer_keeps_original_behavior():
    normalizer = RewardNormalizer(0.97, 5.0, torch.device("cpu"))
    normalizer.update_from_transitions(
        torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([[0.0], [1.0], [0.0]])
    )
    assert normalizer.g_r.item() == pytest.approx(4.94)


def test_done_rows_reset_independently_and_terminal_peak_is_retained():
    normalizer = RewardNormalizer(0.97, 5.0, torch.device("cpu"), episode_boundaries=True)
    normalizer.update_from_transitions(
        torch.tensor([[1.0, 10.0], [2.0, 1.0]]), torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    )
    torch.testing.assert_close(normalizer.g_r, torch.tensor([0.0, 1.0]))
    assert normalizer.g_r_max.item() == 10.0
    assert normalizer.rms.mean.item() == pytest.approx((1 + 10 + 2.97 + 1) / 4.0001)
