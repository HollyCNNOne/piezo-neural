"""Rung 3d 阶段二：完整双缆 MRG（McIntyre 2002）vs HH 对比 + 远场延拓。

与 Rung 3c 的区别：
  1. fiber_type="mrg" 是完整双缆（第一缆轴膜 Vi↔Vp + 第二缆髓鞘 Vp↔φe，静息 -80）。
  2. φe 线只在压电 patch [0,2mm] 内有值（PINN 域），两端是回电流负凹（-0.061/-0.585 V）
     没衰减到 0。真实神经纤维比 patch 长，patch 在纤维中间、两侧 φe→0。这里用三次 Hermite
     把 φe 平滑延拓到 0（匹配边界值+斜率，远端值/斜率→0），再把 patch 放到纤维正中间，
     消除「纤维密封端恰好落在场强区」的边界假象。

性能：HH 用粗网格(nx_cable=101, dx=60µm << 0.2mm 特征)求解，MRG 用精细场网格(301)插值。
二者连续极限下 dx 相消（见 _solve_hh 的 g_a·lap 与 C_m·A_m 归一），阈值与 dx 无关。

用法（服务器 pzneuro env）：
    cd /root/autodl-tmp/pzneuro
    python -u run_rung3d_compare.py --csv outputs/rung3b_full/phi_e_line.csv --a-um 0.5
    python -u run_rung3d_compare.py --csv ... --a-um 1.7   # 标准 MRG 直径(轴突 3.4µm)
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pzneuro.coupling.activating_function import activating_function_1d
from pzneuro.coupling.gain import physical_gain
from pzneuro.neural.cable import solve_cable

p = argparse.ArgumentParser()
p.add_argument("--csv", default="outputs/rung3b_full/phi_e_line.csv")
p.add_argument("--nx", type=int, default=301, help="精细场网格点数（f 报告 + MRG φe 插值）")
p.add_argument("--nx-cable", type=int, default=101, help="HH 粗求解网格点数（dx<<0.2mm 特征即可）")
p.add_argument("--pad-mm", type=float, default=2.0, help="patch 每侧延拓长度 mm")
p.add_argument("--t-end", type=float, default=20.0)
p.add_argument("--a-um", type=float, default=0.5, help="轴半径 µm（HH/MRG 同用）")
p.add_argument("--out", default="outputs/rung3d")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

# ---------------- 读 φe（patch 内，x∈[0,2mm]） ----------------
d = np.loadtxt(args.csv, delimiter=",", skiprows=1)
x_m, phi_V = d[:, 0], d[:, 1]          # m, V


def _extend(x, phi, pad):
    """三次 Hermite 把场平滑延拓到 0：两侧各加 pad 长度。
    交界处匹配(值, 斜率)，远端(值, 斜率)→0。返回 (x_ext, phi_ext)。"""
    x0, p0 = x[0], phi[0]
    sl0 = (phi[1] - phi[0]) / (x[1] - x[0])
    xl = np.linspace(x0 - pad, x0, 50, endpoint=False)
    t = (xl - (x0 - pad)) / pad
    pl = p0 * (-2 * t ** 3 + 3 * t ** 2) + (sl0 * pad) * (t ** 3 - t ** 2)
    x1, p1 = x[-1], phi[-1]
    sl1 = (phi[-1] - phi[-2]) / (x[-1] - x[-2])
    xr = np.linspace(x1, x1 + pad, 50, endpoint=True)[1:]
    t = (xr - x1) / pad
    pr = p1 * (2 * t ** 3 - 3 * t ** 2 + 1) + (sl1 * pad) * (t ** 3 - 2 * t ** 2 + t)
    return np.concatenate([xl, x, xr]), np.concatenate([pl, phi, pr])


pad = args.pad_mm * 1e-3
xe, pe = _extend(x_m, phi_V, pad)
xe = xe - xe[0]                        # patch 移到纤维正中间（MRG 从 x[0] 起建隔室）

# 精细场网格（f 报告 + MRG φe 插值）
x_fine = np.linspace(xe[0], xe[-1], args.nx) * 100.0
phi_fine = np.interp(x_fine, xe * 100.0, pe * 1000.0)
# 平滑窗口取物理长度 ~0.22mm（与 Rung 3c 一致），不随域长变
dx_mm = (x_fine[-1] - x_fine[0]) / (args.nx - 1) * 10.0   # dx 单位 mm
window = max(5, int(round(0.22 / dx_mm)) | 1)             # 保证奇数
phi_s = savgol_filter(phi_fine, window, 3)
f = activating_function_1d(x_fine, phi_s)
print("纤维域 x∈[%.2f, %.2f] cm，patch 占 [%.2f, %.2f] cm"
      % (x_fine[0], x_fine[-1], pad * 100, (pad + (x_m[-1] - x_m[0])) * 100))
print("φe 峰值=%.1f mV | f 正瓣=%.3e @x=%.4f cm, 负瓣=%.3e @x=%.4f cm"
      % (phi_s.max(), f.max(), x_fine[int(f.argmax())], f.min(), x_fine[int(f.argmin())]))
print("φe 两端=%.3f / %.3f mV（延拓后应≈0）" % (phi_s[0], phi_s[-1]))

# 粗电缆网格（HH 求解）
x_cable = np.linspace(x_fine[0], x_fine[-1], args.nx_cable)
phi_cable = np.interp(x_cable, x_fine, phi_s)

gain_phys = physical_gain()["gain"]
print("\n物理增益 gain_phys = %.3e，φe_axon(峰值) = %.1f mV"
      % (gain_phys, phi_s.max() * gain_phys))


def _solve(ft, gain, dt_eval=0.02):
    if ft == "hh":
        x, ph = x_cable, phi_cable
    else:
        x, ph = x_fine, phi_s
    return solve_cable(x, ph * gain, a_um=args.a_um, fiber_type=ft,
                       t_span=(0.0, args.t_end), dt_eval=dt_eval,
                       rtol=1e-4, atol=1e-4)


# ---------------- HH vs MRG（双缆） ----------------
rows = {}
print("\n=== HH(无髓) vs MRG(有髓双缆) @gain_phys ===")
for ft in ["hh", "mrg"]:
    r = _solve(ft, gain_phys)
    rows[ft] = r
    V = r["V"]; i_fire = int(np.argmax(np.max(V, axis=1)))
    kind = ("节点" if r["is_node"][i_fire] else "结间") if ft == "mrg" else "隔室"
    print("  %-4s V_max=%7.1f mV  fired=%s  @x=%.4f cm (%s)"
          % (ft.upper(), V.max(), bool(V.max() > 0.0), r["x"][i_fire], kind))

# ---------------- 阈值扫描 ----------------
print("\n=== 放电阈值 gain 扫描（对数二分） ===")
def _fires(g, ft):
    return _solve(ft, g, dt_eval=0.05)["V"].max() > 0.0

thresh, fire_loc = {}, {}
for ft in ["hh", "mrg"]:
    lo, hi = 1e-4, 1e-2
    for _ in range(14):
        mid = np.sqrt(lo * hi)
        if _fires(mid, ft):
            hi = mid
        else:
            lo = mid
    thr = np.sqrt(lo * hi)
    thresh[ft] = thr
    r = _solve(ft, thr)
    i_fire = int(np.argmax(np.max(r["V"], axis=1)))
    fire_loc[ft] = r["x"][i_fire]
    print("  %-4s 阈值 gain=%.3e (φe_axon=%.1f mV) 放电@x=%.4f cm"
          % (ft.upper(), thr, thr * phi_s.max(), fire_loc[ft]))

# ---------------- 图 ----------------
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ft, axrow in zip(["hh", "mrg"], axes):
    r = rows[ft]
    ax1, ax2 = axrow
    im = ax1.imshow(r["V"], aspect="auto", origin="lower",
                    extent=[r["t"][0], r["t"][-1], r["x"][0], r["x"][-1]],
                    cmap="RdBu_r", vmin=-90, vmax=50)
    ax1.set_xlabel("t (ms)"); ax1.set_ylabel("x (cm)")
    ax1.set_title("%s V_m(x,t)  gain_phys=%.2e (a=%.1f µm)"
                  % (ft.upper(), gain_phys, args.a_um))
    fig.colorbar(im, ax=ax1)
    i_fire = int(np.argmax(np.max(r["V"], axis=1)))
    ax2.plot(r["t"], r["V"][i_fire])
    ax2.set_xlabel("t (ms)"); ax2.set_ylabel("V_m (mV)")
    lab = ("节点" if r["is_node"][i_fire] else "结间") if ft == "mrg" else "隔室"
    ax2.set_title("%s 放电%s V_m(t) @x=%.4f cm" % (ft.upper(), lab, r["x"][i_fire]))
plt.tight_layout()
tag = "a%.1f" % args.a_um
plt.savefig(os.path.join(args.out, "hh_vs_mrg_doublecable_%s.png" % tag), dpi=150)
plt.close()

print("\n完成：图见 %s/hh_vs_mrg_doublecable_%s.png" % (args.out, tag))
