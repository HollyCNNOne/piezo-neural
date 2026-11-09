"""Hodgkin-Huxley 膜（无髓鞘纤维通道），标准 1952 参数（mV 平移版）。

单位系（与 cable.py 一致）：V[mV]、t[ms]、电流密度[µA/cm²]、电导[mS/cm²]、
电容[µF/cm²]。自洽：mS·mV = µA，µF·mV/ms = µA。

    i_ion = g_Na m³h (V-E_Na) + g_K n⁴ (V-E_K) + g_L (V-E_L)
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

HH_PARAMS = dict(g_Na=120.0, g_K=36.0, g_L=0.3,    # mS/cm²
                 E_Na=50.0, E_K=-77.0, E_L=-54.4)  # mV
V_REST = -65.0  # mV，静息电位（初值/参考）


def _alpha_m(V): return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
def _beta_m(V):  return 4.0 * np.exp(-(V + 65.0) / 18.0)
def _alpha_h(V): return 0.07 * np.exp(-(V + 65.0) / 20.0)
def _beta_h(V):  return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
def _alpha_n(V): return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
def _beta_n(V):  return 0.125 * np.exp(-(V + 65.0) / 80.0)


def ion_current(V, m, h, n, p=None):
    """离子电流密度 [µA/cm²]。V,m,h,n 可标量或数组（广播）。"""
    p = p or HH_PARAMS
    return (p["g_Na"] * m ** 3 * h * (V - p["E_Na"])
            + p["g_K"] * n ** 4 * (V - p["E_K"])
            + p["g_L"] * (V - p["E_L"]))


def gate_rhs(V, m, h, n):
    """门变量导数 (dm/dt, dh/dt, dn/dt) [1/ms]。"""
    am, bm = _alpha_m(V), _beta_m(V)
    ah, bh = _alpha_h(V), _beta_h(V)
    an, bn = _alpha_n(V), _beta_n(V)
    return (am * (1 - m) - bm * m,
            ah * (1 - h) - bh * h,
            an * (1 - n) - bn * n)


def steady_state(V):
    """V 处的平衡门变量 (m_inf, h_inf, n_inf)。"""
    return (_alpha_m(V) / (_alpha_m(V) + _beta_m(V)),
            _alpha_h(V) / (_alpha_h(V) + _beta_h(V)),
            _alpha_n(V) / (_alpha_n(V) + _beta_n(V)))


def solve_single_hh(I_ext, t_span=(0.0, 100.0), y0=None, t_eval=None,
                    params=None):
    """0 维 HH（单隔室，无空间）：C_m dV/dt = -i_ion + I_ext。

    I_ext 标量（恒流 µA/cm²）或函数 I_ext(t)。返回 (t, V)。
    用于验证膜动力学 + 作为「无空间点神经元」对照（对应 FHN 的升级）。
    """
    p = params or HH_PARAMS
    if y0 is None:
        m0, h0, n0 = steady_state(V_REST)
        y0 = (V_REST, m0, h0, n0)
    I = (lambda t: I_ext) if np.isscalar(I_ext) else I_ext

    def rhs(t, y):
        V, m, h, n = y
        i_ion = ion_current(V, m, h, n, p)
        dm, dh, dn = gate_rhs(V, m, h, n)
        return [-i_ion / 1.0 + I(t), dm, dh, dn]  # C_m=1 µF/cm² -> /1

    if t_eval is None:
        t_eval = np.arange(t_span[0], t_span[1], 0.01)
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method="BDF",
                    rtol=1e-6, atol=1e-6)
    return sol.t, sol.y[0]
