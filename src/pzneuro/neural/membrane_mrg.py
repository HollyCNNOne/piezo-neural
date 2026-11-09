"""MRG (McIntyre-Richardson-Grill 2002) 有髓轴突膜 —— 双缆(double-cable)版。

节点膜电流（快钠 Na_f + 持久钠 Na_p + 慢钾 K_s + 漏）：
    i_ion = g_Naf m³h (V-E_Na) + g_Nap p³ (V-E_Na) + g_Ks s (V-E_K) + g_Lk (V-E_Lk)
门控 α/β 形式 dx/dt = α(1-x) - β x（Q10=1，36°C）。

参考：McIntyre, Richardson & Grill, J Neurophysiol 87:995-1006 (2002)。
α_m 用 NEURON 参考实现的 21.4（论文印刷值 20.4，Grill 2010 确认实现值更正确）。

【双缆结构 vs Rung3c 单缆塌缩】
  第一缆（轴膜，轴浆 ↔ 轴周间隙）：
    节点   —— 有源通道（Na_f/Na_p/K_s/漏），电容 2 µF/cm²
    结间   —— 被动漏电 g_pas=0.0001 S/cm²（=0.1 mS/cm²）、E_pas=-80，电容 2 µF/cm²
  第二缆（髓鞘，轴周间隙 ↔ 细胞外）：
    N=80 片层串联，每片两层膜 → 有效电容/电导 = 单片值 /(N·2)
    单片 C_my=0.1 µF/cm²、G_my=0.001 S/cm²（=1 mS/cm²）→ 有效 6.25e-4 µF/cm²、6.25e-3 mS/cm²

  Rung3c 单缆版把「结间轴膜 + 髓鞘」塌缩成一个被动 RC(E_PAS=-90)，两个错：
  ① 漏了髓鞘屏障（真髓鞘有效电导 6.25e-3 mS/cm² ≪ 轴膜 0.1 mS/cm²）；
  ② 静息反电位 -90 应为 -80。双缆恢复真实结间 R-C 网络，静息自然回到 ~-80。
  具体双电位 ODE 见 cable._solve_mrg。

单位系（与 cable.py / membrane_hh.py 一致）：V[mV]、t[ms]、电流[µA/cm²]、
电导[mS/cm²]、电容[µF/cm²]。MRG 原文用 S/cm²，这里 ×1000 转 mS/cm²。
"""
from __future__ import annotations
import numpy as np

# 电导原文 3.0 / 0.01 / 0.08 / 0.007 S/cm²，×1000 → mS/cm²
MRG_PARAMS = dict(g_Naf=3000.0, g_Nap=10.0, g_Ks=80.0, g_Lk=7.0,
                  E_Na=50.0, E_K=-90.0, E_Lk=-90.0)   # mV（漏与钾同反电位）

# 第一缆：轴膜（节点有源 / 结间被动）
C_NODE = 2.0          # µF/cm² 节点轴膜电容
C_AXOLEMMA = 2.0      # µF/cm² 结间轴膜电容（与节点同膜，McIntyre ci=2）
G_AXOLEMMA = 0.1      # mS/cm² 结间轴膜被动漏电导（=0.0001 S/cm²，McIntyre g_pas）
E_AXOLEMMA = -80.0    # mV 结间轴膜被动漏电反电位（McIntyre e_pas=v_init=-80）

# 第二缆：髓鞘（N 片层串联，每片两层膜 → 有效值 /(N·2)）
N_LAM = 80
C_MY_LAYER = 0.1      # µF/cm² 单片髓鞘电容
G_MY_LAYER = 1.0      # mS/cm² 单片髓鞘漏电导（=0.001 S/cm²）
E_MY = -80.0          # mV 髓鞘漏电反电位
C_MY_EFF = C_MY_LAYER / (N_LAM * 2.0)   # ≈6.25e-4 µF/cm²
G_MY_EFF = G_MY_LAYER / (N_LAM * 2.0)   # ≈6.25e-3 mS/cm²

# 轴周间隙（结间轴膜与髓鞘之间的薄环隙）—— 第二缆的轴向通道
RHO_PA = 70.0         # Ω·cm 轴周间隙电阻率（McIntyre 轴浆/轴周 70 Ω·cm）
GAP_PA = 6.4e-3       # µm 轴周间隙宽度（≈6.4 nm）


def _rate(z, A, k):
    """线性形式速率 A·z/(1-exp(-z/k))，z→0 时极限 A·k。"""
    z = np.asarray(z, dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        r = A * z / (1.0 - np.exp(-z / k))
    return np.where(np.abs(z) < 1e-9, A * k, r)


def _rate_alpha_m(V): return _rate(V + 21.4, 6.57, 10.3)
def _rate_beta_m(V):  return _rate(-(V + 25.7), 0.304, 9.16)
def _rate_alpha_h(V): return _rate(-(V + 114.0), 0.34, 11.0)
def _rate_beta_h(V):  return 12.6 / (1.0 + np.exp(-(V + 31.8) / 13.4))
def _rate_alpha_p(V): return _rate(V + 27.0, 0.0353, 10.2)
def _rate_beta_p(V):  return _rate(-(V + 34.0), 0.000883, 10.0)
def _rate_alpha_s(V): return 0.3 / (1.0 + np.exp(-(V + 53.0) / 5.0))
def _rate_beta_s(V):  return 0.03 / (1.0 + np.exp(-(V + 90.0) / 1.0))


def mrg_ion_current(V, m, h, p, s, is_node, params=None):
    """离子电流密度 [µA/cm²]，向量化。is_node=True 用有源节点轴膜，False 用结间被动轴膜。

    注意：V 是「轴膜电位」= V_i - V_pa（结间）或 V_i - V_e（节点，cable 里 V_pa 已置为 φe）。
    门变量物理上 ∈[0,1]、V ∈[-200,100] mV，松容差 BDF 解 g_Naf=3000 刚系统时状态会瞬时
    越界致 m**3 溢出，这里对局部副本 clip 防呆，不改求解器状态，不影响正常放电轨迹。
    """
    prm = params or MRG_PARAMS
    m = np.clip(m, 0.0, 1.0); h = np.clip(h, 0.0, 1.0)
    p = np.clip(p, 0.0, 1.0); s = np.clip(s, 0.0, 1.0)
    V = np.clip(V, -200.0, 100.0)
    i_node = (prm["g_Naf"] * m ** 3 * h * (V - prm["E_Na"])
              + prm["g_Nap"] * p ** 3 * (V - prm["E_Na"])
              + prm["g_Ks"] * s * (V - prm["E_K"])
              + prm["g_Lk"] * (V - prm["E_Lk"]))
    i_pas = G_AXOLEMMA * (V - E_AXOLEMMA)
    return np.where(is_node, i_node, i_pas)


def mrg_gate_rhs(V, m, h, p, s, is_node):
    """门变量导数 (dm, dh, dp, ds) [1/ms]，向量化。结间隔室门冻结（乘 is_node）。"""
    am, bm = _rate_alpha_m(V), _rate_beta_m(V)
    ah, bh = _rate_alpha_h(V), _rate_beta_h(V)
    ap, bp = _rate_alpha_p(V), _rate_beta_p(V)
    as_, bs = _rate_alpha_s(V), _rate_beta_s(V)
    return ((am * (1.0 - m) - bm * m) * is_node,
            (ah * (1.0 - h) - bh * h) * is_node,
            (ap * (1.0 - p) - bp * p) * is_node,
            (as_ * (1.0 - s) - bs * s) * is_node)


def mrg_steady_state(V):
    """V 处的平衡门变量 (m_inf, h_inf, p_inf, s_inf)。"""
    am, bm = _rate_alpha_m(V), _rate_beta_m(V)
    ah, bh = _rate_alpha_h(V), _rate_beta_h(V)
    ap, bp = _rate_alpha_p(V), _rate_beta_p(V)
    as_, bs = _rate_alpha_s(V), _rate_beta_s(V)
    return (am / (am + bm), ah / (ah + bh), ap / (ap + bp), as_ / (as_ + bs))


def rest_potential(lo=-100.0, hi=-50.0):
    """节点本征静息（单个孤立节点，无结间耦合）的 i_ion(V, gates_inf(V))=0 根，≈-61 mV。

    注意：这是「单节点」静息（比 -80 去极化，因快钠窗口电流强）。整根纤维的静息 ~-80
    由结间轴膜被动漏电(E_AXOLEMMA=-80)主导钉住，见 cable._solve_mrg 双缆平衡。
    """
    def net(V):
        m, h, p, s = mrg_steady_state(np.array([V]))
        return mrg_ion_current(np.array([V]), m, h, p, s, np.array([True]))[0]

    l, r = lo, hi
    for _ in range(100):
        mid = 0.5 * (l + r)
        if net(mid) < 0.0:
            l = mid
        else:
            r = mid
    return 0.5 * (l + r)


V_REST_MRG = -80.0   # mV 纤维静息电位（双缆下 ≈ 结间轴膜漏电反电位 E_AXOLEMMA=-80）

# 说明：双缆版里结间轴膜漏电反电位 E_AXOLEMMA=-80（McIntyre e_pas），节点持久钠窗口电流
# 会把整根纤维静息从 -80 轻微抬到 ~-78（结间面积 ≫ 节点面积，抬升很小）。
# 单缆版曾把结间塌缩成 E_PAS=-90 → 静息 -90，是已知局限；双缆已修正。
