"""物理耦合增益：体导体理论标定 φe 从薄膜表面到轴突的衰减（消除 magic 1e-3）。

Rung 3b 里 gain=1e-3 是集总占位。这里用「体导体 + 容性源 AC 耦合」把它算出来。

【物理图景】压电薄膜开路电压 V_oc≈39V 是 Thevenin 电压，但源阻抗是薄膜电容
C_film=ε_film·A/H（高阻容性源，不是理想电压源，DC 下相当于断路）。组织是导电体
（σ_ext），薄膜通过组织向轴突注入电流时，看到的负载是「扩散电阻」
R_spread=ρ_ext/(2πd)（半空间点源）。轴突处电位是容性源对扩散电阻的分压，AC 频率 ω 下：
    gain = |R_spread / (R_spread + 1/(jω C_film))| ≈ ω·C_film·R_spread   （弱耦合极限）
         = 2π·f_stim · C_film · ρ_ext/(2π·d)
         = f_stim · C_film / (σ_ext · d)                                   (1)

三条第一性结论（都从 (1) 直接读出）：
  ① 增益 ∝ 频率 f_stim —— DC/准静态压电不耦合神经，必须 AC/超声刺激；
  ② 增益 ∝ 1/σ_ext   —— 组织越导电衰减越狠（σ_ext 在分母）；
  ③ 增益 ∝ 1/d       —— 间隙越大衰减越狠。
这三条正是用户点名的「体导体 + σ_ext + 组织厚度 d」那一路，替换 magic 1e-3。

默认 f_stim=200kHz（超声压电神经刺激典型频段），代入得 gain≈9e-4，与 magic 1e-3
同量级——原来那个 magic number 恰好对应「超声频段容性耦合」，现在有公式依据。

单位：SI 输入（µm→m），输出无量纲 gain。
"""
from __future__ import annotations
import numpy as np

EPS0 = 8.854187817620e-12  # F/m


def physical_gain(d_um=50.0, a_contact_um=200.0, H_um=200.0,
                  eps_film_rel=12.0, sigma_ext=0.3, f_stim=200e3):
    """体导体 + 容性源 AC 耦合增益，返回 dict(gain, C_film, R_spread, ...)。

    参数：
      d_um            组织间隙厚度（薄膜表面→轴突），典型 50µm
      a_contact_um    压痕/接触半径，Rung 3b 高斯压痕 σ=L/10=200µm
      H_um            薄膜厚度 200µm
      eps_film_rel    PVDF 相对介电常数 ≈12
      sigma_ext       组织电导率 S/m，灰质 ≈0.3（用户指定）
      f_stim          刺激频率 Hz，超声压电刺激 ≈200kHz

    返回 gain = 2π·f_stim·C_film·R_spread（无量纲），C_film 薄膜源电容(F)，
    R_spread 半空间扩散电阻(Ω)。
    """
    d = d_um * 1e-6
    a = a_contact_um * 1e-6
    H = H_um * 1e-6
    A_contact = np.pi * a ** 2                      # m² 接触面积
    C_film = eps_film_rel * EPS0 * A_contact / H    # F 薄膜源电容
    R_spread = 1.0 / (2.0 * np.pi * sigma_ext * d)  # Ω 半空间点源扩散电阻
    gain = 2.0 * np.pi * f_stim * C_film * R_spread  # 无量纲
    return dict(gain=gain, C_film=C_film, R_spread=R_spread,
                d_um=d_um, a_contact_um=a_contact_um, H_um=H_um,
                eps_film_rel=eps_film_rel, sigma_ext=sigma_ext,
                f_stim=f_stim)


def capacitive_divider_gain(d_um=50.0, H_um=200.0, eps_film_rel=12.0,
                            eps_tissue_rel=400.0):
    """【对照/废弃】纯容性分压，把组织当纯电介质 ε_tissue，得 g_cap≈7e-3。

    问题：它把组织当 ε=400 的绝缘体，漏掉了组织导电性 σ_ext 和刺激频率 f_stim。
    真正的组织是导电体、薄膜是容性源，DC 下容性源被组织短路 → 增益必须
    ∝ ω·C_film·R_spread（见 physical_gain）。留作「为什么不能用纯容性分压」的对照。
    """
    d = d_um * 1e-6
    H = H_um * 1e-6
    eps_f = eps_film_rel * EPS0
    eps_t = eps_tissue_rel * EPS0
    return eps_f * d / (eps_f * d + eps_t * H)


def resistive_spread_gain(a_contact_um=200.0, d_um=50.0):
    """Rattay 体导体「几何扩散」增益（与 σ_ext 无关，σ_ext 在比值中约掉）。

    圆盘接触(半径 a)向半空间注入电流：表面电位 φ_surface=I·ρe/(4a)，
    距离 d 处点源电位 φ(d)=I·ρe/(2πd)。比值 gain=φ(d)/φ_surface=2a/(πd)。

    这是纯几何扩散。d=50µm、a=200µm 得 ~2.5（近场不衰减），说明「1000× 衰减」
    不可能来自几何扩散，必须来自薄膜容性源阻抗 × 组织扩散电阻（physical_gain）。
    """
    a = a_contact_um * 1e-6
    d = d_um * 1e-6
    return 2.0 * a / (np.pi * d)
