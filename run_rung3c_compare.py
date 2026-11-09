"""Rung 3c：物理标定 gain + HH(无髓) vs MRG(有髓) 双通道阈值/放电位置对比。

阶段一：用体导体理论把 magic gain=1e-3 换成 physical_gain(d, ε_tissue, ...)。
阶段二：同一非均匀 f(x) 下对比无髓 HH 与有髓 MRG 的放电阈值与放电位置。

只跑 cable 段（读 φe CSV，不重训 PINN），纯 numpy/scipy 可反复快扫。

用法（服务器 pzneuro env）：
    cd /root/autodl-tmp/pzneuro
    python -u run_rung3c_compare.py --csv outputs/rung3b_full/phi_e_line.csv
    python -u run_rung3c_compare.py --csv ... --d-um 100 --eps-tissue 2000
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pzneuro.coupling.activating_function import activating_function_1d
from pzneuro.coupling.gain import (physical_gain, capacitive_divider_gain,
                                   resistive_spread_gain)
from pzneuro.neural.cable import solve_cable

p = argparse.ArgumentParser()
p.add_argument("--csv", default="outputs/rung3b_full/phi_e_line.csv")
p.add_argument("--nx", type=int, default=101, help="φe 重采样点数（f 用）")
p.add_argument("--t-end", type=float, default=20.0)
p.add_argument("--a-um", type=float, default=0.5, help="纤维半径 µm（HH/MRG 同用）")
p.add_argument("--d-um", type=float, default=50.0, help="组织间隙厚度 µm")
p.add_argument("--sigma-ext", type=float, default=0.3, help="组织电导率 S/m（灰质≈0.3）")
p.add_argument("--f-stim", type=float, default=200e3, help="刺激频率 Hz（超声压电≈200kHz）")
p.add_argument("--out", default="outputs/rung3c")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)

# ---------------- 读 φe，重采样 + 平滑 + 激活函数 f ----------------
d = np.loadtxt(args.csv, delimiter=",", skiprows=1)
x_m, phi_V = d[:, 0], d[:, 1]
x_cm = np.linspace(x_m[0], x_m[-1], args.nx) * 100.0
phi_mV = np.interp(x_cm, x_m * 100.0, phi_V * 1000.0)
window = max(5, 2 * (len(x_cm) // 20) + 1)
phi_s = savgol_filter(phi_mV, window, 3)
f = activating_function_1d(x_cm, phi_s)
ipk = int(f.argmax())
print("φe peak=%.1f mV | f +瓣=%.1f @x=%.4f cm, -瓣=%.1f @x=%.4f cm"
      % (phi_s.max(), f.max(), x_cm[ipk], f.min(), x_cm[int(f.argmin())]))

# ---------------- 阶段一：物理增益（替换 magic 1e-3） ----------------
pg = physical_gain(d_um=args.d_um, sigma_ext=args.sigma_ext, f_stim=args.f_stim)
gain_phys = pg["gain"]
print("\n=== 阶段一：物理增益（替换 magic 1e-3） ===")
print("  薄膜源电容 C_film = %.3e F  (ε_film=%.0f, A=π(%.0fµm)², H=%.0fµm)"
      % (pg["C_film"], pg["eps_film_rel"], pg["a_contact_um"], pg["H_um"]))
print("  扩散电阻 R_spread = %.3e Ω  (σ_ext=%.2f S/m, d=%.0fµm, 半空间 1/2πσd)"
      % (pg["R_spread"], pg["sigma_ext"], pg["d_um"]))
print("  gain = 2π·f_stim·C_film·R_spread = %.3e  (f_stim=%.0f kHz)"
      % (gain_phys, pg["f_stim"] / 1e3))
print("  【对照】纯容性分压(组织当电介质) = %.3e —— 漏了 σ_ext 与频率，高估 ~7×"
      % capacitive_divider_gain(d_um=args.d_um))
print("  【对照】几何扩散 2a/(πd) = %.2f —— 近场 O(1)，1000× 来自源阻抗非几何"
      % resistive_spread_gain(a_contact_um=200.0, d_um=args.d_um))
print("  φe 峰值 %.1f mV × gain_phys = %.1f mV（轴突处细胞外电位）"
      % (phi_s.max(), phi_s.max() * gain_phys))

# ---------------- 阶段二：HH vs MRG ----------------
print("\n=== 阶段二：HH(无髓) vs MRG(有髓) 双通道 ===")
fibers = ["hh", "mrg"]
rows = {}
for ft in fibers:
    r = solve_cable(x_cm, phi_s * gain_phys, a_um=args.a_um, fiber_type=ft,
                    t_span=(0.0, args.t_end), dt_eval=0.02, rtol=1e-4, atol=1e-4)
    rows[ft] = r
    Vmax = r["V"].max()
    fired = bool(Vmax > 0.0)
    i_fire = int(np.argmax(np.max(r["V"], axis=1)))
    x_fire = r["x"][i_fire]
    if ft == "mrg":
        kind = "节点" if r["is_node"][i_fire] else "结间"
        print("  %-4s V_max=%7.1f mV  fired=%s  放电隔室 x=%.4f cm (%s)"
              % (ft.upper(), Vmax, fired, x_fire, kind))
    else:
        print("  %-4s V_max=%7.1f mV  fired=%s  放电隔室 x=%.4f cm"
              % (ft.upper(), Vmax, fired, x_fire))

# ---------------- 阈值扫描（对数二分，精确到 ~1%） ----------------
print("\n=== 放电阈值 gain 扫描（HH vs MRG，对数二分） ===")
def _fires(g, ft):
    r = solve_cable(x_cm, phi_s * g, a_um=args.a_um, fiber_type=ft,
                    t_span=(0.0, args.t_end), dt_eval=0.02, rtol=1e-4, atol=1e-4)
    return r["V"].max() > 0.0

thresh = {}
for ft in fibers:
    lo, hi = 1e-4, 1e-2
    for _ in range(14):
        mid = np.sqrt(lo * hi)
        if _fires(mid, ft):
            hi = mid
        else:
            lo = mid
    thr = np.sqrt(lo * hi)
    thresh[ft] = thr
    print("  %-4s 阈值 gain ≈ %.3e  (φe_axon=%.1f mV, 相对 φe_peak %g)"
          % (ft.upper(), thr, thr * phi_s.max(), thr))

# ---------------- 图 ----------------
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ft, axrow in zip(fibers, axes):
    r = rows[ft]
    ax1, ax2 = axrow
    im = ax1.imshow(r["V"], aspect="auto", origin="lower",
                    extent=[r["t"][0], r["t"][-1], r["x"][0], r["x"][-1]],
                    cmap="RdBu_r", vmin=-90, vmax=50)
    ax1.set_xlabel("t (ms)"); ax1.set_ylabel("x (cm)")
    ax1.set_title("%s  V_m(x,t)  gain_phys=%.2e" % (ft.upper(), gain_phys))
    fig.colorbar(im, ax=ax1)
    i_fire = int(np.argmax(np.max(r["V"], axis=1)))
    ax2.plot(r["t"], r["V"][i_fire])
    ax2.set_xlabel("t (ms)"); ax2.set_ylabel("V_m (mV)")
    lab = "节点" if (ft == "mrg" and r["is_node"][i_fire]) else "隔室"
    ax2.set_title("%s 放电%s V_m(t) @x=%.4f cm" % (ft.upper(), lab, r["x"][i_fire]))
plt.tight_layout()
plt.savefig(os.path.join(args.out, "hh_vs_mrg.png"), dpi=150)
plt.close()

print("\n=== 完成，图见 %s/hh_vs_mrg.png ===" % args.out)
