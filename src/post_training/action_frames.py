"""Convert absolute TCP action chunks using the measured pose at their anchor."""
import numpy as np


def rotation6d(values):
    values = np.asarray(values, dtype=float)
    a, b = values[..., :3], values[..., 3:]
    length = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(length < 1e-6):
        raise ValueError("Degenerate TCP rotation")
    x = a / length
    y = b - np.sum(x*b, axis=-1, keepdims=True)*x
    length = np.linalg.norm(y, axis=-1, keepdims=True)
    if np.any(length < 1e-6):
        raise ValueError("Degenerate TCP rotation")
    y /= length
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def relative_tcp_actions(actions, reference, valid):
    """Layout: left xyz/rot6d, right xyz/rot6d, two motor coordinates [rad]."""
    actions = np.asarray(actions, np.float32).copy()
    reference = np.asarray(reference, np.float32)
    if actions.ndim != 2 or actions.shape[1] != 20 or reference.shape != (18,):
        raise ValueError("Absolute TCP conversion requires action[H,20] and reference[18]")
    if not np.isfinite(reference).all():
        raise ValueError("TCP reference must be finite")
    for offset in (0, 9):
        eligible = valid[:, offset:offset+9]
        rows = np.all(eligible, axis=-1)
        if np.any(np.any(eligible, axis=-1) != rows):
            raise ValueError("TCP translation/rotation must be valid or masked together")
        ref = reference[offset:offset+9]
        rotation = rotation6d(ref[3:])
        target = actions[rows, offset:offset+9].copy()
        target[:, :3] = (target[:, :3] - ref[:3]) @ rotation
        relative = rotation.T @ rotation6d(target[:, 3:])
        target[:, 3:] = np.concatenate([relative[..., :, 0], relative[..., :, 1]], axis=-1)
        actions[rows, offset:offset+9] = target
    return actions
