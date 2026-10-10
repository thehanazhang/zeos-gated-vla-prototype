import numpy as np
import pytest
from zeos_vla.display_mapping import display_targets, LOWER, NEUTRAL, UPPER


def test_display_mapping_preserves_neutral_and_joint_limits():
    initial = np.arange(7, dtype=np.float32)
    chunk = np.tile(initial, (16, 1))
    np.testing.assert_allclose(display_targets(chunk, initial), np.tile(NEUTRAL, (16, 1)))
    chunk[1:] += 1000
    targets = display_targets(chunk, initial)
    assert targets.shape == (16, 8)
    assert np.all(targets >= LOWER) and np.all(targets <= UPPER)
    assert np.any(targets[1] != targets[0])


def test_display_mapping_rejects_bad_inputs_before_dispatch():
    with pytest.raises(ValueError):
        display_targets(np.zeros((7, 16)), np.zeros(7))
    chunk = np.zeros((16, 7))
    chunk[5, 1] = np.nan
    with pytest.raises(ValueError):
        display_targets(chunk, np.zeros(7))
