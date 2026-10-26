"""Checkpoint copies retain noncanonical bool scratch bytes, not only truth values."""

import pytest
import torch

from unilab.managers.continuation import copy_tree, restore_tree


@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
def test_bool_scratch_copy_and_restore_preserve_original_bytes(shape):
    saved = torch.tensor([31], dtype=torch.uint8).reshape(shape).view(torch.bool)
    copied = copy_tree(saved)
    assert torch.equal(copied.view(torch.uint8), saved.view(torch.uint8))
    target = torch.zeros(shape, dtype=torch.bool)
    pointer = target.data_ptr()
    actual = restore_tree(target, saved, device="cpu")
    assert actual is target and actual.data_ptr() == pointer
    assert torch.equal(actual.view(torch.uint8), saved.view(torch.uint8))
    allocated = restore_tree(None, saved, device="cpu")
    assert torch.equal(allocated.view(torch.uint8), saved.view(torch.uint8))
    assert allocated.data_ptr() != saved.data_ptr()


def test_strided_bool_restore_preserves_neighboring_storage_and_every_byte():
    saved = torch.tensor([0, 1, 7, 255], dtype=torch.uint8).view(torch.bool)
    storage = torch.full((8,), 13, dtype=torch.uint8)
    target = storage.view(torch.bool)[::2]
    assert not target.is_contiguous()
    assert restore_tree(target, saved, device="cpu") is target
    assert torch.equal(storage[::2], saved.view(torch.uint8))
    assert torch.equal(storage[1::2], torch.full((4,), 13, dtype=torch.uint8))
    assert torch.equal(copy_tree(target).view(torch.uint8), saved.view(torch.uint8))
