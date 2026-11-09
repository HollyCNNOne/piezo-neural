"""方形域 [0,L] x [0,H] 的采样。

以后加 disk.py / pointcloud.py 时实现同样的接口即可，训练代码不动。
"""
from __future__ import annotations
import numpy as np


def sample_interior(n, L, H, generator=None, seed=0):
    rng = np.random.default_rng(seed) if generator is None else generator
    x = rng.uniform(0.0, L, n)
    z = rng.uniform(0.0, H, n)
    return np.stack([x, z], axis=1)


def boundary_points(n_per_edge, L, H, generator=None, seed=0):
    """返回各边界上的点。键：left / right / top / bottom。"""
    rng = np.random.default_rng(seed) if generator is None else generator
    t = rng.uniform(0.0, 1.0, n_per_edge)
    return {
        "left":   np.stack([np.full_like(t, 0.0), t * H], axis=1),
        "right":  np.stack([np.full_like(t, L), t * H], axis=1),
        "top":    np.stack([t * L, np.full_like(t, H)], axis=1),
        "bottom": np.stack([t * L, np.full_like(t, 0.0)], axis=1),
    }
