"""Rung 3d 第一步：验证 HH 无髓纤维在 x=0.10(负瓣/超极化区) 的阳极断开(anode break)机制。

画双 Y 轴 V_m(t)+h(t) @x=0.10，标注 spike 时间，检查 h 在强超极化期是否去失活(h↑)，
并对比相邻隔室(0.09/0.11) 看空间虚拟阳极效应（邻室去极化电流经轴向电阻拉起中心）。

用法（服务器 pzneuro env）：
    cd /root/autodl-tmp/pzneuro
    python -u run_rung3d_diagnose.py --csv outputs/rung3b_full/phi_e_line.csv
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pzneuro.coupling.gain import physical_gain
from pzneuro.neural.cable import solve_cable

p = argparse.ArgumentParser()
p.add_argument("--csv", default="outputs/rung3b_full/phi_e_line.csv")
p.add_argument("--nx", type=int, default=101)
p.add_argument("--t-end", type=float, default=20.0)
p.add_argument("--a-um", type=float, default=0.5)
p.add_argument("--out", default="outputs/rung3d")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

# 读 φe + 平滑 + gain（与 Rung 3c 完全一致）
d = np.loadtxt(args.csv, delimiter=",", skiprows=1)
x_m, phi_V = d[:, 0], d[:, 1]
x_cm = np.linspace(x_m[0], x_m[-1], args.nx) * 100.0
phi_mV = np.interp(x_cm, x_m * 100.0, phi_V * 1000.0)
window = max(5, 2 * (len(x_cm) // 20) + 1)
phi_s = savgol_filter(phi_mV, window, 3)
gain = physical_gain()["gain"]

# HH 解（带门变量）
r = solve_cable(x_cm, phi_s * gain, a_um=args.a_um, fiber_type="hh",
                t_span=(0.0, args.t_end), dt_eval=0.02, rtol=1e-4, atol=1e-4)
t, V, h, x = r["t"], r["V"], r["h"], r["x"]


def idx(x0):
    return int(np.argmin(np.abs(x - x0)))


i0 = idx(0.10)
iL = idx(0.09)
iR = idx(0.11)
print("目标隔室 x=%.4f cm (i=%d)，邻 0.09(i=%d)/0.11(i=%d)"
      % (x[i0], i0, iL, iR))

V0, h0 = V[i0], h[i0]
spk_mask = V0 > 0.0
if spk_mask.any():
    t_spike = t[spk_mask][np.argmax(V0[spk_mask])]
    pre = t < t_spike
    t_hyper = t[pre][np.argmin(V0[pre])]
    h_at_hyper = h0[pre][np.argmin(V0[pre])]
    h_at_spike = h0[np.argmin(np.abs(t - t_spike))]
else:
    t_spike = None
    t_hyper = t[np.argmin(V0)]
    h_at_hyper = h0[np.argmin(V0)]
    h_at_spike = np.nan

print("超极化谷 V_min=%.1f mV @t=%.2f ms，该时刻 h=%.3f（去失活程度）"
      % (V0.min(), t_hyper, h_at_hyper))
if t_spike is not None:
    print("spike V_max=%.1f mV @t=%.2f ms，该时刻 h=%.3f"
          % (V0.max(), t_spike, h_at_spike))
    print("=> h 从超极化期 %.3f → spike 时 %.3f（去失活后重新激活）"
          % (h_at_hyper, h_at_spike))
else:
    print("=> 未放电")

# 图 1：双 Y 轴 V_m + h
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax1.plot(t, V0, color="tab:blue", lw=1.5, label="V_m")
ax1.axhline(0, color="k", lw=0.5, ls=":")
ax1.set_ylabel("V_m (mV)", color="tab:blue")
ax1.set_title("x=%.2f cm HH 阳极断开诊断 (gain=%.2e)" % (x[i0], gain))
if t_spike is not None:
    ax1.axvline(t_spike, color="tab:red", ls="--", lw=1)
    ax1.annotate("spike t=%.2f ms" % t_spike, xy=(t_spike, V0.max()),
                 xytext=(t_spike + 0.8, V0.max() - 15), color="tab:red",
                 arrowprops=dict(arrowstyle="->", color="tab:red"))
ax1.axvline(t_hyper, color="tab:green", ls="--", lw=0.8)
ax1.annotate("hyper t=%.2f ms\nh=%.2f" % (t_hyper, h_at_hyper),
             xy=(t_hyper, V0.min()), xytext=(max(t_hyper - 5, 0), V0.min() + 25),
             color="tab:green", fontsize=8)

ax1b = ax1.twinx()
ax1b.plot(t, h0, color="tab:orange", lw=1.5, label="h (Na inactivation)")
ax1b.set_ylabel("h gate", color="tab:orange")
ax1b.set_ylim(0, 1)
ax1b.tick_params(axis="y", labelcolor="tab:orange")
lines = ax1.get_lines() + ax1b.get_lines()
ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right")

# 图 2：相邻隔室（空间虚拟阳极）
ax2.plot(t, V[iL], label="x=0.09 cm", lw=1.2)
ax2.plot(t, V[i0], label="x=0.10 cm (目标)", lw=2)
ax2.plot(t, V[iR], label="x=0.11 cm", lw=1.2)
ax2.axhline(0, color="k", lw=0.5, ls=":")
ax2.set_xlabel("t (ms)")
ax2.set_ylabel("V_m (mV)")
ax2.set_title("相邻隔室 V_m(t)：邻室去极化经轴向电阻拉起中心(空间虚拟阳极)")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(args.out, "anode_break_diagnosis.png"), dpi=150)
plt.close()
print("\n图见 %s/anode_break_diagnosis.png" % args.out)
