"""BODY_NAMES 27-body contract (aligns with GH real training data).

GH's body_names_keep (utils/motion.py:105) lists "world", but select_in_order(...,
return_missing=False) (motion.py:115) drops names not present in the retargeted input,
and the GMR G1 model has no "world" body — so GH's real memmap is 27-body (pelvis first),
and the memmap dimension is data-driven (len(meta['body_names']), motion.py:258). Phase 3
mis-copied the 28-name whitelist as a hardcoded expectation; this locks the real 27.
"""
from unilab.envs.gh_tracking.motion_dataset import BODY_NAMES, FIELD_SPEC, NUM_BODIES


def test_body_names_is_27_starting_at_pelvis_no_world():
    assert "world" not in BODY_NAMES
    assert BODY_NAMES[0] == "pelvis"
    assert len(BODY_NAMES) == 27
    assert NUM_BODIES == 27


def test_field_spec_body_dims_follow_27():
    # body_pos_w / body_pos_b -> (27, 3); body_quat_w -> (27, 4)
    assert FIELD_SPEC["body_pos_w"][1] == (27, 3)
    assert FIELD_SPEC["body_pos_b"][1] == (27, 3)
    assert FIELD_SPEC["body_quat_w"][1] == (27, 4)
