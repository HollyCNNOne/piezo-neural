"""PINN 训练循环。"""
from __future__ import annotations
import numpy as np
import torch

from ..geometry.square import sample_interior, boundary_points
from .losses import loss as loss_primal


def train(model, mat, sigma0, L, H, *, epochs=8000, n_int=2000, n_bc=200,
          lr=1e-3, seed=0, log_every=500, weights=None, device=None,
          lr_decay=False, loss_fn=None, lbfgs=False, lbfgs_steps=300):
    """Adam 训练，可选余弦 LR 衰减、可选 LBFGS 精调（固定采样点）。

    lr_decay=True  余弦退火，压振荡（Rung 2a）。
    lbfgs=True     Adam 收敛后换 LBFGS 精调，对付病态曲面（Rung 2c）。
    loss_fn        可换 8 场混合格式损失。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = loss_fn or loss_primal
    rng = np.random.default_rng(seed)
    hist = []

    # ---- Adam 主训练 ----
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = None
    if lr_decay:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr * 1e-3)
    for ep in range(1, epochs + 1):
        xy_int = sample_interior(n_int, L, H, generator=rng)
        bc = boundary_points(n_bc, L, H, generator=rng)
        total, comps = loss_fn(model, xy_int, bc, mat, sigma0, L, H, weights)
        opt.zero_grad()
        total.backward()
        opt.step()
        if sched is not None:
            sched.step()
        hist.append(float(total.detach().cpu()))
        if ep % log_every == 0:
            print("epoch %6d/%d  loss=%.3e  (pde=%.3e, bc=%.3e)"
                  % (ep, epochs, hist[-1], comps["pde"], comps["bc"]))

    # ---- LBFGS 精调（固定一批点，标准 PINN 做法） ----
    if lbfgs:
        xy_int = sample_interior(n_int * 2, L, H, generator=rng)
        bc = boundary_points(n_bc * 2, L, H, generator=rng)
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                                history_size=100,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            total, _ = loss_fn(model, xy_int, bc, mat, sigma0, L, H, weights)
            total.backward()
            return total

        for s in range(1, lbfgs_steps + 1):
            total = opt.step(closure)
            hist.append(float(total.detach().cpu()))
            if s == 1 or s % 20 == 0:
                print("lbfgs %4d/%d  loss=%.3e" % (s, lbfgs_steps, hist[-1]))
    return hist
