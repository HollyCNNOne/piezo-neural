"""Rung 3b：端到端「力(局部压痕) -> 电(PINN 非均匀 φe) -> 神经(cable+HH)」空间选择性预测。

与 run_baseline.py 的区别：顶面载荷从均匀 szz=-sigma0 换成局部高斯压痕
    szz = -F0 * exp(-(x-x_center)^2 / (2 sigma^2))
从而 φe(x) 沿 x 变化 -> 激活函数 f(x)=d²φe/dx² 非零 -> cable 产生空间选择性放电
（去极化瓣在 x≈x_center±√3σ、中心超极化，经典 Rattay 双瓣）。

用法（服务器 pzneuro env）：
    cd /root/autodl-tmp/pzneuro
    python run_rung3b_pipeline.py                       # 局部压痕 + 8 场混合
    python run_rung3b_pipeline.py --sigma 1e-4          # 更窄压痕
    python run_rung3b_pipeline.py --lbfgs               # Adam 后 LBFGS 精调
    python run_rung3b_pipeline.py --epochs 2000 --tag qk # 快速跑（CPU 被占时）

输出 ./outputs/rung3b[_<tag>]/：
    phi_field.png    PINN φ 场（局部压痕下电势集中）
    phi_e_f.png      φe(x) 与 f(x) 对比（应见「双瓣」）
    cable_V.png      V_m(x,t) 时空热图 + 放电隔室 V_m(t)
    phi_e_line.csv   φe(x) 顶面电势（SI：m, V）
"""
import os, sys, argparse
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pzneuro.physics.materials import PVDF
from pzneuro.models.mlp_mixed import MLPMixed
from pzneuro.pde.losses import characteristic_scales
from pzneuro.pde.losses_mixed import loss_mixed
from pzneuro.pde.train import train
from pzneuro.coupling.activating_function import (eval_phi_line, save_phi_line,
                                                  activating_function_1d)
from pzneuro.neural.cable import solve_cable

# ------------------ 问题配置 ------------------
p = argparse.ArgumentParser(description="Rung 3b 力->电->神经 空间选择性 pipeline")
p.add_argument("--L", type=float, default=2.0e-3, help="横向尺寸 m")
p.add_argument("--H", type=float, default=2.0e-4, help="厚度 m")
p.add_argument("--sigma0", type=float, default=1.0e6, help="参考应力 Pa")
p.add_argument("--x-center", type=float, default=None, help="压痕中心 m（默认 L/2）")
p.add_argument("--sigma", type=float, default=None, help="压痕高斯宽度 m（默认 L/10）")
p.add_argument("--F0", type=float, default=None, help="压痕峰值应力 Pa（默认 sigma0）")
p.add_argument("--epochs", type=int, default=4000)
p.add_argument("--lbfgs", action="store_true", help="Adam 后 LBFGS 精调")
p.add_argument("--lbfgs-steps", type=int, default=100)
p.add_argument("--nx", type=int, default=201, help="φe(x) 采样点数")
p.add_argument("--no-smooth", action="store_true", help="关闭 φe 的 savgol 平滑")
p.add_argument("--gains", type=str, default="1e-4,1e-3,3e-3,1e-2",
               help="φe 耦合增益（组织衰减集总参数），逗号分隔")
p.add_argument("--tag", default="")
args = p.parse_args()

MAT = PVDF
L, H = args.L, args.H
SIGMA0 = args.sigma0
x_center = args.x_center if args.x_center is not None else L / 2.0
sigma = args.sigma if args.sigma is not None else L / 10.0
F0 = args.F0 if args.F0 is not None else SIGMA0
load = {"x_center": x_center, "sigma": sigma, "F0": F0}
gains = [float(g) for g in args.gains.split(",")]

OUTDIR = os.path.join("outputs", "rung3b_" + args.tag) if args.tag else os.path.join("outputs", "rung3b")
os.makedirs(OUTDIR, exist_ok=True)

print("=== Rung 3b: L=%.3e H=%.3e sigma0=%.3e, 压痕(xc=%.3e, sigma=%.3e, F0=%.3e) ==="
      % (L, H, SIGMA0, x_center, sigma, F0))

# ------------------ PINN 训练（局部压痕载荷） ------------------
sc = characteristic_scales(MAT, SIGMA0, L, H)
torch.manual_seed(0); np.random.seed(0)
model = MLPMixed(u_scale=sc["Ux"], v_scale=sc["Uz"], phi_scale=sc["PHI"],
                 sigma_scale=sc["SIGMA"], d_scale=sc["D"], L=L, H=H,
                 hidden=64, layers=4)
loss_fn = partial(loss_mixed, load=load)
print("网络参数量:", sum(p.numel() for p in model.parameters()))

train(model, MAT, SIGMA0, L, H, epochs=args.epochs, n_int=2000, n_bc=200,
      lr=1e-3, seed=0, loss_fn=loss_fn, lbfgs=args.lbfgs,
      lbfgs_steps=args.lbfgs_steps)

# ------------------ Step 2：提取顶面 φe(x) ------------------
z_axon = H                          # 轴突贴顶面（开路电极侧，电位最高）
x_cm, phi_e_mV = eval_phi_line(model, L, H, z_axon=z_axon, nx=args.nx)
save_phi_line(model, L, H, z_axon, os.path.join(OUTDIR, "phi_e_line.csv"), nx=args.nx)
print("φe(x): peak=%.1f mV @x=%.3f cm, min=%.1f mV"
      % (phi_e_mV.max(), x_cm[phi_e_mV.argmax()], phi_e_mV.min()))

# 数值微分前平滑：f=d²φe/dx² 是二阶导，对 φe 噪声极敏感；savgol 是标准做法，不改物理
phi_use = phi_e_mV
if not args.no_smooth:
    window = max(5, 2 * (len(x_cm) // 20) + 1)
    phi_use = savgol_filter(phi_e_mV, window, 3)

# ------------------ Step 3：激活函数 f(x) ------------------
f = activating_function_1d(x_cm, phi_use)   # mV/cm²
ipk, imn = int(f.argmax()), int(f.argmin())
print("f(x): +瓣=%.1f mV/cm² @x=%.3f cm, -瓣=%.1f @x=%.3f cm（理论 ±√3σ≈±%.3f cm）"
      % (f.max(), x_cm[ipk], f.min(), x_cm[imn], 3 ** 0.5 * sigma * 100))

fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
a1.plot(x_cm, phi_use, ".-", ms=3); a1.set_ylabel("φe (mV)")
a1.set_title("顶面电势 φe(x)（局部压痕）")
a2.plot(x_cm, f, ".-", ms=3); a2.axhline(0, color="k", lw=0.5)
a2.set_ylabel("f = d²φe/dx² (mV/cm²)"); a2.set_xlabel("x (cm)")
a2.set_title("激活函数（双瓣：两侧去极化、中心超极化）")
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "phi_e_f.png"), dpi=150); plt.close()

# φ 场图（PINN 自身，无解析解可比——均匀解的 exact_solution 已不适用）
nx2 = nz2 = 101
xs = np.linspace(0, L, nx2); zs = np.linspace(0, H, nz2)
X, Z = np.meshgrid(xs, zs)
xy = np.stack([X.ravel(), Z.ravel()], axis=1)
with torch.no_grad():
    phi2 = model(torch.tensor(xy, dtype=torch.float32,
                              device=next(model.parameters()).device))[:, 2].cpu().numpy()
phi2 = phi2.reshape(nz2, nx2)
fig, ax = plt.subplots(figsize=(7, 3))
c = ax.contourf(X, Z, phi2, 60, cmap="RdBu_r")
ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_title("PINN φ field（局部压痕）")
fig.colorbar(c, ax=ax)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "phi_field.png"), dpi=150); plt.close()

# ------------------ Step 3：cable + HH（耦合增益扫阈值） ------------------
print("\n=== cable 放电（0.5 µm 无髓纤维，增益扫阈值） ===")
results = []
for g in gains:
    r = solve_cable(x_cm, phi_use * g, t_span=(0.0, 20.0), dt_eval=0.02)
    fired = bool(r["V"].max() > 0.0)
    results.append((g, r, fired))
    print("  gain=%.0e  V_max=%7.1f mV  fired=%s" % (g, r["V"].max(), fired))

firing = [(g, r) for g, r, fd in results if fd]
if firing:
    g_fire, r_fire = firing[0]
    t, V = r_fire["t"], r_fire["V"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    im = a1.imshow(V, aspect="auto", origin="lower",
                   extent=[t[0], t[-1], x_cm[0], x_cm[-1]],
                   cmap="RdBu_r", vmin=-80, vmax=50)
    a1.set_xlabel("t (ms)"); a1.set_ylabel("x (cm)"); a1.set_title("V_m(x,t) gain=%.0e" % g_fire)
    fig.colorbar(im, ax=a1)
    a2.plot(t, V[ipk]); a2.set_xlabel("t (ms)"); a2.set_ylabel("V_m (mV)")
    a2.set_title("放电隔室 V_m(t) @x=%.3f cm（f 去极化瓣）" % x_cm[ipk])
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "cable_V.png"), dpi=150); plt.close()
    print("\n用 gain=%.0e 出 cable 图（放电隔室 x=%.3f cm，V 峰 %.1f mV）"
          % (g_fire, x_cm[ipk], V.max()))
else:
    print("\n警告：给定增益范围内 cable 均未放电（试调大 --gains）")

print("\n=== 完成，输出见 ./%s/ ===" % OUTDIR)
