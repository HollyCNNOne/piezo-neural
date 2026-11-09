"""FitzHugh-Nagumo 神经动力学（scipy 求解）。

    dv/dt = v - v^3/3 - w + I_ext
    dw/dt = eps*(v + a - b*w)

经典参数 a=0.7, b=0.8, eps=0.08，外电流 I_ext 约 > 0.34 时进入持续放电（Hopf）。
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp


def solve_fhn(I_ext, t_span=(0.0, 200.0), y0=(-1.0, 0.0), params=None, t_eval=None):
    """求解 FHN，返回 (t, v, w)。

    I_ext 可以是标量（恒定电流）或函数 I_ext(t)。
    """
    p = {"a": 0.7, "b": 0.8, "epsilon": 0.08}
    if params:
        p.update(params)
    I = (lambda t: I_ext) if np.isscalar(I_ext) else I_ext

    def rhs(t, y):
        v, w = y
        return [v - v ** 3 / 3.0 - w + I(t), p["epsilon"] * (v + p["a"] - p["b"] * w)]

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 4000)
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method="RK45",
                    rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[0], sol.y[1]


def detect_spikes(t, v, threshold=0.0):
    """数 v 上升沿穿过 threshold 的次数（约等于放电次数）。"""
    above = v > threshold
    return int(np.count_nonzero(above[1:] & ~above[:-1]))
