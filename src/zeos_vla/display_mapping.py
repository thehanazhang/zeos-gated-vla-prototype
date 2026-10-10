"""Illustrative simulation scaling, deliberately not a B601 calibration."""

import numpy as np

JOINT_NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper_joint1", "gripper_joint2"]
NEUTRAL = np.array([0, 0.8, 1.0, 0, 0, 0, 0.025, 0.03575], dtype=np.float32)
LOWER = np.array([-2.8, 0, 0, -1.57, -1.57, -3.14, 0, 0], dtype=np.float32)
UPPER = np.array([2.8, 3.14, 3.14, 1.57, 1.57, 3.14, 0.05, 0.0715], dtype=np.float32)


def display_targets(actions, initial_proprio):
    """Map recorded-unit deltas around a neutral pose; clip to URDF limits.

    Arm scale is an arbitrary 0.05 radians per recorded unit. Gripper scale is
    an arbitrary 1% of each slider's travel per recorded unit, shared by both
    fingers. Neither scale represents the recording driver's motor conversion.
    """
    actions = np.asarray(actions, dtype=np.float32)
    initial = np.asarray(initial_proprio, dtype=np.float32)
    if actions.shape != (16, 7) or initial.shape != (7,):
        raise ValueError("Expected a 16x7 chunk and seven recorded state values")
    if not np.isfinite(actions).all() or not np.isfinite(initial).all():
        raise ValueError("Non-finite display input")
    delta = actions - initial
    targets = np.tile(NEUTRAL, (16, 1))
    targets[:, :6] += 0.05 * delta[:, :6]
    targets[:, 6:] += delta[:, 6:7] * (0.01 * UPPER[6:])
    return np.clip(targets, LOWER, UPPER)
