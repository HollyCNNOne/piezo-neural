"""一键 baseline：力 → 电（PINN 正演）→ 神经放电（FHN）。

用法（在有 torch 的服务器上）：
    cd pzneuro
    python run_baseline.py                              # Rung1 薄片 2mm x 200um，3场
    python run_baseline.py --H 2e-3 --tag square        # 诊断：拉成 1:1 正方形
    python run_baseline.py --lr-decay --tag lrdecay     # Rung2a：加余弦 LR 衰减
    python run_baseline.py --mixed --tag mixed          # Rung2b：8 场混合格式

输出到 ./outputs/<tag>/：
    loss_history.png   训练损失曲线
    phi_field.png      电势场 PINN vs 解析解
    fhn_spikes.png     不同刺激强度下的 FHN 动作电位（阈值演示）
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pzneuro.physics.materials import PVDF
from pzneuro.physics.constitutive import exact_solution
from pzneuro.models.mlp import MLP
from pzneuro.models.mlp_mixed import MLPMixed
from pzneuro.pde.losses import characteristic_scales, loss as loss_primal
from pzneuro.pde.losses_mixed import loss_mixed
from pzneuro.pde.train import train
from pzneuro.pde.evaluate import evaluate, plot_fields
from pzneuro.neural.fhn import solve_fhn, detect_spikes
from pzneuro.coupling.stimulus import step_current

# ------------------ 问题配置（消融实验用命令行开关切换） ------------------
p = argparse.ArgumentParser(description="力->电(PINN)->神经(FHN) baseline")
p.add_argument("--L", type=float, default=2.0e-3, help="横向尺寸 m")
p.add_argument("--H", type=float, default=2.0e-4, help="厚度 m")
p.add_argument("--sigma0", type=float, default=1.0e6, help="顶面压缩应力 Pa")
p.add_argument("--epochs", type=int, default=8000)
p.add_argument("--lr-decay", action="store_true", help="余弦 LR 衰减（Rung 2a）")
p.add_argument("--mixed", action="store_true", help="8 场混合格式（Rung 2b）")
p.add_argument("--lbfgs", action="store_true", help="Adam 后 LBFGS 精调（Rung 2c）")
p.add_argument("--lbfgs-steps", type=int, default=300)
p.add_argument("--tag", default="", help="输出子目录名，区分消融实验")
args = p.parse_args()

MAT = PVDF          # 材料（模块一实测后替换 PVDF 的 5 个数）
L, H = args.L, args.H
SIGMA0 = args.sigma0

# 训练超参
EPOCHS = args.epochs
N_INT, N_BC = 2000, 200
LR = 1e-3
SEED = 0
HIDDEN, LAYERS = 64, 4

OUTDIR = os.path.join("outputs", args.tag) if args.tag else "outputs"
os.makedirs(OUTDIR, exist_ok=True)

print("=== 配置: L=%.3e m, H=%.3e m, sigma0=%.3e, epochs=%d, lr_decay=%s, mixed=%s, tag=%r ==="
      % (L, H, SIGMA0, EPOCHS, args.lr_decay, args.mixed, args.tag))

# ------------------ 特征尺度（输出缩放 + 损失归一化共用） ------------------
sc = characteristic_scales(MAT, SIGMA0, L, H)
print("特征尺度: Ux=%.3e m, Uz=%.3e m, PHI=%.3e V, SIGMA=%.3e Pa, D=%.3e C/m^2"
      % (sc["Ux"], sc["Uz"], sc["PHI"], sc["SIGMA"], sc["D"]))

# ------------------ 模块三：PINN 正演 ------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
if args.mixed:
    model = MLPMixed(u_scale=sc["Ux"], v_scale=sc["Uz"], phi_scale=sc["PHI"],
                     sigma_scale=sc["SIGMA"], d_scale=sc["D"], L=L, H=H,
                     hidden=HIDDEN, layers=LAYERS)
    loss_fn = loss_mixed
else:
    model = MLP(u_scale=sc["Ux"], v_scale=sc["Uz"], phi_scale=sc["PHI"],
                L=L, H=H, hidden=HIDDEN, layers=LAYERS)
    loss_fn = loss_primal
print("网络参数量:", sum(p.numel() for p in model.parameters()))

hist = train(model, MAT, SIGMA0, L, H, epochs=EPOCHS, n_int=N_INT,
             n_bc=N_BC, lr=LR, seed=SEED, lr_decay=args.lr_decay,
             loss_fn=loss_fn, lbfgs=args.lbfgs, lbfgs_steps=args.lbfgs_steps)

plt.figure(figsize=(5, 3.5))
plt.semilogy(hist)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("training loss")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "loss_history.png"), dpi=150)
plt.close()

# ------------------ 验证：与解析解对比 ------------------
X, Z, u_p, v_p, phi_p, u_e, v_e, phi_e, errs = evaluate(model, MAT, SIGMA0, L, H)
print("\n=== 与解析解的相对 L2 误差（<1e-2 视为跑通） ===")
for k, e in errs.items():
    print("  %-4s: %.4e" % (k, e))

phi_top_pinn = float(phi_p[-1, phi_p.shape[1] // 2])
phi_top_exact = float(phi_e[-1, phi_e.shape[1] // 2])
print("  phi_top(PINN)=%.4e V, phi_top(解析)=%.4e V" % (phi_top_pinn, phi_top_exact))

plot_fields(X, Z, phi_p, phi_e, outdir=OUTDIR)

# ------------------ 模块四：电势 -> FHN 神经放电 ------------------
phi_ref = abs(phi_top_exact) if phi_top_exact != 0 else 1.0
phi_norm = phi_top_pinn / phi_ref  # 归一化到 ~[0,1]

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, g_c in zip(axes, [0.1, 0.3, 0.5]):   # g_c 是占位耦合常数（量级待标定）
    I0 = g_c * phi_norm
    t, v, w = solve_fhn(step_current(I0, t_on=20.0), t_span=(0.0, 200.0),
                        y0=(-1.0, 0.0))
    n_spk = detect_spikes(t, v)
    ax.plot(t, v)
    ax.set_title("g_c=%.1f, I0=%.3f, spikes=%d" % (g_c, I0, n_spk))
    ax.set_xlabel("t")
    ax.set_ylabel("v (膜电位)")
fig.suptitle("压电电势 -> FHN 神经放电（阈值演示）")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fhn_spikes.png"), dpi=150)
plt.close()

print("\n=== 完成，输出见 ./%s/ ===" % OUTDIR)
