"""电势 -> 刺激电流（耦合接口）。

物理近似（写进论文 Methods，明说是 lumped 近似）：
把细胞膜-膜接触区看成一个集总导纳，I_ext = g_c * (phi - phi_rest)。
g_c 是唯一量级不确定的集总参数，后续敏感性分析扫它。

baseline 里用归一化电势 phi_norm = phi/phi_ref ∈ [0,1]，g_c 直接给注入电流幅值。
"""
from __future__ import annotations


def normalized_potential(phi, phi_ref):
    return phi / phi_ref


def step_current(I0, t_on=20.0):
    """阶跃刺激：t < t_on 为 0，之后恒为 I0。"""
    return lambda t: I0 if t >= t_on else 0.0
