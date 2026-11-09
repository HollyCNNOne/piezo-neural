"""只跑 cable 部分：读已保存的 φe CSV，算 f，扫增益出放电。不重训 PINN。

训练一旦出 CSV，cable 段就是纯 numpy/scipy，可单独反复快速扫参数。

用法：
    cd /root/autodl-tmp/pzneuro
    python run_cable_only.py --csv outputs/rung3b_full/phi_e_line.csv
    python run_cable_only.py --csv ... --gains 1e-3,3e-3 --nx 101
"""
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
from scipy.signal import savgol_filter

from pzneuro.coupling.activating_function import activating_function_1d
from pzneuro.neural.cable import solve_cable

p = argparse.ArgumentParser()
p.add_argument("--csv", default="outputs/rung3b_full/phi_e_line.csv")
p.add_argument("--nx", type=int, default=101, help="cable 隔室数")
p.add_argument("--gains", default="1e-4,1e-3,3e-3,1e-2")
p.add_argument("--t-end", type=float, default=20.0)
args = p.parse_args()

d = np.loadtxt(args.csv, delimiter=",", skiprows=1)
x_m, phi_V = d[:, 0], d[:, 1]

# 重采样到 nx 点，转神经单位 cm / mV
x_cm = np.linspace(x_m[0], x_m[-1], args.nx) * 100.0
phi_mV = np.interp(x_cm, x_m * 100.0, phi_V * 1000.0)

# savgol 平滑（二阶导前）
window = max(5, 2 * (len(x_cm) // 20) + 1)
phi_s = savgol_filter(phi_mV, window, 3)

f = activating_function_1d(x_cm, phi_s)
ipk, imn = int(f.argmax()), int(f.argmin())
print("φe 重采样后 peak=%.1f mV | f +瓣=%.1f @x=%.4f cm, -瓣=%.1f @x=%.4f cm"
      % (phi_s.max(), f.max(), x_cm[ipk], f.min(), x_cm[imn]))

for g in [float(v) for v in args.gains.split(",")]:
    r = solve_cable(x_cm, phi_s * g, t_span=(0.0, args.t_end), dt_eval=0.02)
    print("gain=%.0e  V_max=%7.1f mV  fired=%s"
          % (g, r["V"].max(), bool(r["V"].max() > 0.0)))
