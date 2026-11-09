"""在网格上评估 + 与解析解对比 + 出图。"""
from __future__ import annotations
import os
import numpy as np
import torch

from ..physics.constitutive import exact_solution


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def evaluate(model, mat, sigma0, L, H, nx=101, nz=101):
    """返回 (X, Z, u_p, v_p, phi_p, u_e, v_e, phi_e, errs)。"""
    ex = exact_solution(mat, sigma0, L, H)
    xs = np.linspace(0.0, L, nx)
    zs = np.linspace(0.0, H, nz)
    X, Z = np.meshgrid(xs, zs)
    xy = np.stack([X.ravel(), Z.ravel()], axis=1)
    with torch.no_grad():
        out = model(torch.tensor(xy, dtype=torch.float32,
                                 device=next(model.parameters()).device)).cpu().numpy()
    u_p = out[:, 0].reshape(nx, nz)
    v_p = out[:, 1].reshape(nx, nz)
    phi_p = out[:, 2].reshape(nx, nz)
    u_e, v_e, phi_e = ex["u"](X, Z), ex["v"](X, Z), ex["phi"](X, Z)
    errs = {"u": _rel_l2(u_p, u_e), "v": _rel_l2(v_p, v_e),
            "phi": _rel_l2(phi_p, phi_e)}
    return X, Z, u_p, v_p, phi_p, u_e, v_e, phi_e, errs


def plot_fields(X, Z, phi_pred, phi_exact, outdir="outputs", fname="phi_field.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    vmax = max(float(np.max(np.abs(phi_exact))), 1e-30)
    c0 = axes[0].contourf(X, Z, phi_exact, 40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("phi exact")
    c1 = axes[1].contourf(X, Z, phi_pred, 40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("phi PINN")
    c2 = axes[2].contourf(X, Z, np.abs(phi_pred - phi_exact), 40, cmap="viridis")
    axes[2].set_title("|error|")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal")
    fig.colorbar(c0, ax=axes[0])
    fig.colorbar(c1, ax=axes[1])
    fig.colorbar(c2, ax=axes[2])
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)
