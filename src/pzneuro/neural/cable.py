"""一维隔室 cable 方程 + 细胞外激活函数源项（电->神经 空间耦合核心）。

离散式（隔室 n，膜面积 A_n、左右轴向电导 g_a^L/g_a^R）：
    C_n A_n dV_n/dt = g_a^L (V_{n-1}-V_n) + g_a^R (V_{n+1}-V_n) - A_n i_ion(V_n)
                      + g_a^L (φe_{n-1}-φe_n) + g_a^R (φe_{n+1}-φe_n)
均匀网格退化为经典形式（A_m=2πa·dx、g_a=πa²/(R_i·dx)）：
    C_m A_m dV_n/dt = g_a (V_{n-1}-2V_n+V_{n+1}) - A_m i_ion + g_a (φe_{n-1}-2φe_n+φe_{n+1})
最后一项即激活函数 f=d²φe/dx² 的源项贡献（Rattay 1986 现代标准用法）。

fiber_type:
  "hh"  无髓鞘：整根纤维均匀有源 HH 膜（1952）。
  "mrg" 有髓鞘：节点(MRG 有源膜) + 结间(轴膜被动+髓鞘) 交替，双缆（McIntyre 2002）。

单位系：cm、ms、mV、µA、µF、mS、kΩ（自洽：mS·mV=µA，µF·mV/ms=µA）。
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

from .membrane_hh import (ion_current as hh_ion, gate_rhs as hh_gates,
                          steady_state as hh_steady, V_REST)
from .membrane_mrg import (mrg_ion_current, mrg_gate_rhs, mrg_steady_state,
                           V_REST_MRG, C_NODE, C_AXOLEMMA, C_MY_EFF, G_MY_EFF,
                           RHO_PA, GAP_PA)


def _lap_1d(y):
    """一维离散拉普拉斯（反射/密封边界）。"""
    lap = np.empty_like(y)
    lap[1:-1] = y[:-2] - 2.0 * y[1:-1] + y[2:]
    lap[0] = y[1] - y[0]
    lap[-1] = y[-2] - y[-1]
    return lap


def solve_cable(x, phi_e, *, a_um=0.5, R_i_ohm_cm=35.4, C_m=1.0,
                t_span=(0.0, 50.0), dt_eval=0.01, I_app=None, params=None,
                fiber_type="hh", node_len_um=1.0, inter_um=None,
                rtol=1e-6, atol=1e-6):
    """解 cable，返回 dict{t, V, x, f, phi_e, ...}。

    x, phi_e 同长：x 单位 cm、phi_e 单位 mV。fiber_type 见模块 docstring。
    mrg 时额外返回 x_comp(隔室中心)、is_node(节点掩码)、A、C。
    I_app 可选注入电流 [µA/cm²]（标量或函数；通常 None 表示只靠细胞外场驱动）。
    rtol/atol：BDF 求解容差。MRG 节点 g_Naf=3000 很刚，阈值扫描可松到 1e-4 加速
    （只判是否放电，不需解析尖峰形状）。
    """
    x = np.asarray(x, float)
    phi_e = np.asarray(phi_e, float)
    if fiber_type == "mrg":
        return _solve_mrg(x, phi_e, a_um=a_um, R_i_ohm_cm=R_i_ohm_cm,
                          node_len_um=node_len_um, inter_um=inter_um,
                          t_span=t_span, dt_eval=dt_eval, I_app=I_app,
                          rtol=rtol, atol=atol)
    return _solve_hh(x, phi_e, a_um=a_um, R_i_ohm_cm=R_i_ohm_cm, C_m=C_m,
                     t_span=t_span, dt_eval=dt_eval, I_app=I_app, params=params,
                     rtol=rtol, atol=atol)


def _solve_hh(x, phi_e, *, a_um, R_i_ohm_cm, C_m, t_span, dt_eval, I_app, params,
              rtol, atol):
    N = x.size
    dx = (x[-1] - x[0]) / (N - 1)
    a = a_um * 1e-4                 # µm -> cm
    R_i = R_i_ohm_cm / 1000.0       # Ω·cm -> kΩ·cm
    A_m = 2.0 * np.pi * a * dx      # cm²，每隔室膜面积
    g_a = np.pi * a ** 2 / (R_i * dx)  # mS，轴向电导

    # 细胞外强迫电流 [µA]：g_a * d²φe/dx² 的离散
    f_force = g_a * _lap_1d(phi_e)

    m0, h0, n0 = hh_steady(V_REST)
    V0 = np.full(N, V_REST)
    y0 = np.concatenate([V0, np.full(N, m0), np.full(N, h0), np.full(N, n0)])

    if I_app is None:
        I_app_call = None
    elif np.isscalar(I_app):
        I_app_call = lambda t: I_app
    else:
        I_app_call = I_app

    def rhs(t, y):
        V = y[0:N]
        m = y[N:2 * N]
        h = y[2 * N:3 * N]
        n = y[3 * N:4 * N]
        i_ion = hh_ion(V, m, h, n, params)          # µA/cm²
        inj = 0.0 if I_app_call is None else I_app_call(t)  # µA/cm²
        dV = (g_a * _lap_1d(V) - A_m * i_ion + f_force + A_m * inj) / (C_m * A_m)
        dm, dh, dn = hh_gates(V, m, h, n)
        return np.concatenate([dV, dm, dh, dn])

    t_eval = np.arange(t_span[0], t_span[1], dt_eval)
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method="BDF",
                    rtol=rtol, atol=atol)
    V = sol.y[0:N, :]
    m = sol.y[N:2 * N, :]
    h = sol.y[2 * N:3 * N, :]
    n = sol.y[3 * N:4 * N, :]

    # 报告激活函数 f(x)=d²φe/dx²（mV/cm²），与神经侧单位一致
    f = np.gradient(np.gradient(phi_e, x), x)
    return {"t": sol.t, "V": V, "x": x, "f": f, "phi_e": phi_e,
            "m": m, "h": h, "n": n}


def _mrg_compartments(L_cm, node_len_um, inter_um):
    """构建有髓纤维隔室：节点(短,有源)+结间(长,无源) 从左到右交替。

    inter_um 是节点中心间距（结间长度），节点本身长 node_len_um，
    故结间隔室长 = inter_um - node_len_um。返回 (centers_cm, lengths_cm, is_node)。
    """
    node_len = node_len_um * 1e-4
    inter = inter_um * 1e-4
    centers, lengths, types = [], [], []
    x = 0.0
    while x < L_cm - 1e-12:
        Ln = min(node_len, L_cm - x)
        centers.append(x + Ln / 2.0)
        lengths.append(Ln)
        types.append(True)
        x += Ln
        if x >= L_cm - 1e-12:
            break
        Li = min(inter - node_len, L_cm - x)
        centers.append(x + Li / 2.0)
        lengths.append(Li)
        types.append(False)
        x += Li
    return (np.array(centers), np.array(lengths), np.array(types, dtype=bool))


def _solve_mrg(x, phi_e, *, a_um, R_i_ohm_cm, node_len_um, inter_um,
               t_span, dt_eval, I_app, rtol, atol):
    """双缆(double-cable) MRG：每隔室两个电位 V_i(轴浆) + V_pa(轴周间隙)。

    第一缆（轴膜，V_i ↔ V_pa）：节点有源 MRG 膜，结间被动漏电 G_AXOLEMMA/E_AXOLEMMA。
    第二缆（髓鞘，V_pa ↔ 细胞外 φe）：N=80 片层串联的有效 G_MY_EFF/C_MY_EFF。
    节点无髓鞘 → V_pa 冻结在 φe（轴膜直接暴露给细胞外场）。
    """
    L_cm = x[-1] - x[0]
    if inter_um is None:
        # 结间长度 = 100 × 纤维直径。纤维直径 = 轴突直径 / g-ratio(0.6)。
        # 之前误用 100 × 轴突直径(2a)，轴向电导 g_a 偏大 1.67×，
        # 节点 Na 电流被结间轴浆泄流吸干 → 阈值高 10×（accommodation）。
        inter_um = 100.0 * (2.0 * a_um) / 0.6
    a = a_um * 1e-4                        # µm -> cm 轴半径
    R_i = R_i_ohm_cm / 1000.0              # Ω·cm -> kΩ·cm 轴浆电阻率
    rho_pa = RHO_PA / 1000.0               # Ω·cm -> kΩ·cm 轴周间隙电阻率
    gap = GAP_PA * 1e-4                    # µm -> cm 轴周间隙宽度

    centers, lengths, is_node = _mrg_compartments(L_cm, node_len_um, inter_um)
    N = centers.size

    # 第一缆：轴膜面积/电容（节点有源 / 结间被动）
    A_a = 2.0 * np.pi * a * lengths                  # cm²
    C_a = np.where(is_node, C_NODE, C_AXOLEMMA)      # µF/cm²（都是 2.0）

    # 第二缆：髓鞘面积/电容（结间有效 R-C；节点无髓鞘）
    a_my = a + gap
    A_my = 2.0 * np.pi * a_my * lengths              # cm²
    C_my = np.full(N, C_MY_EFF)                      # µF/cm²

    # 轴向电导（相邻隔室中心距）
    dx_between = 0.5 * (lengths[:-1] + lengths[1:])
    g_a_between = np.pi * a ** 2 / (R_i * dx_between)              # mS 轴浆轴向
    g_pa_between = 2.0 * np.pi * a * gap / (rho_pa * dx_between)   # mS 轴周间隙轴向
    g_aL = np.concatenate([[0.0], g_a_between]); g_aR = np.concatenate([g_a_between, [0.0]])
    g_paL = np.concatenate([[0.0], g_pa_between]); g_paR = np.concatenate([g_pa_between, [0.0]])

    # 细胞外电位 φe 插值到隔室中心
    phi_c = np.interp(centers, x, phi_e)

    # 初值：V_i=-80（静息），V_pa：节点=φe（直接暴露）、结间=0（静息无场）
    V0 = np.full(N, V_REST_MRG)
    Vp0 = np.where(is_node, phi_c, 0.0)
    m0, h0, p0, s0 = mrg_steady_state(V_REST_MRG)
    y0 = np.concatenate([V0, Vp0, np.full(N, m0), np.full(N, h0),
                         np.full(N, p0), np.full(N, s0)])

    if I_app is None:
        I_app_call = None
    elif np.isscalar(I_app):
        I_app_call = lambda t: I_app
    else:
        I_app_call = I_app

    def rhs(t, y):
        Vi = y[0:N]
        Vp = y[N:2 * N]
        m = y[2 * N:3 * N]; h = y[3 * N:4 * N]
        p = y[4 * N:5 * N]; s = y[5 * N:6 * N]
        Vax = Vi - Vp                                   # 轴膜电位
        i_ax = mrg_ion_current(Vax, m, h, p, s, is_node)  # µA/cm² 轴膜电流(向外)
        inj = 0.0 if I_app_call is None else I_app_call(t)
        # 轴浆轴向
        Vi_l = np.concatenate([[Vi[0]], Vi[:-1]]); Vi_r = np.concatenate([Vi[1:], [Vi[-1]]])
        axial_i = g_aL * (Vi_l - Vi) + g_aR * (Vi_r - Vi)
        # 轴周间隙轴向
        Vp_l = np.concatenate([[Vp[0]], Vp[:-1]]); Vp_r = np.concatenate([Vp[1:], [Vp[-1]]])
        axial_pa = g_paL * (Vp_l - Vp) + g_paR * (Vp_r - Vp)
        # 髓鞘漏电（轴周→细胞外）
        i_my = G_MY_EFF * (Vp - phi_c)
        # 第二缆 ODE：轴周间隙（节点无髓鞘，Vp 冻结在 φe）
        #   C_m d(Vp-φe)/dt = axial_pa + A_a i_ax - A_m i_my（φe 静态 → dφe/dt=0）
        dVp = (axial_pa + A_a * i_ax - A_my * i_my) / (C_my * A_my)
        dVp = np.where(is_node, 0.0, dVp)
        # 第一缆 ODE：轴浆。轴膜电容跨在 Vi 与 Vp 之间：
        #   C_a d(Vi-Vp)/dt = axial_i - A_a i_ax + A_a inj
        #   => dVi/dt = (axial_i - A_a i_ax + A_a inj)/(C_a A_a) + dVp/dt
        dVi = (axial_i - A_a * i_ax + A_a * inj) / (C_a * A_a) + dVp
        dm, dh, dp, ds = mrg_gate_rhs(Vax, m, h, p, s, is_node)
        return np.concatenate([dVi, dVp, dm, dh, dp, ds])

    t_eval = np.arange(t_span[0], t_span[1], dt_eval)
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method="BDF",
                    rtol=rtol, atol=atol)
    Vi = sol.y[0:N, :]
    Vp = sol.y[N:2 * N, :]
    Vax = Vi - Vp      # 轴膜电位（节点=Vi-φe，结间=Vi-Vp）
    m = sol.y[2 * N:3 * N, :]; h = sol.y[3 * N:4 * N, :]
    p = sol.y[4 * N:5 * N, :]; s = sol.y[5 * N:6 * N, :]
    f = np.gradient(np.gradient(phi_e, x), x)
    return {"t": sol.t, "V": Vax, "x": centers, "f": f, "phi_e": phi_e,
            "x_comp": centers, "is_node": is_node, "A": A_a, "C": C_a,
            "Vi": Vi, "Vp": Vp, "m": m, "h": h, "p": p, "s": s}
