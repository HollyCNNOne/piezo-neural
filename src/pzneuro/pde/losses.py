"""PINN 损失：3 场原变量 (u, v, phi) 的残差 + 边界条件。

3 个未知对应 3 条残差：
    平衡：div(sigma) = 0   (2 条)
    高斯：div(D)     = 0   (1 条)

其中 sigma/D 由 (u,v,phi) 经本构 + 自动微分算出。

量纲问题：u ~ sigma0*L/C11, v ~ sigma0*H/C11, phi ~ |g33|*sigma0*H,
sigma ~ sigma0, D ~ |d33|*sigma0，相差多个数量级。所以每个损失项都除以自己的
特征尺度，让所有项 O(1)（与参考代码 PINNs_piezoelectricity-main 的 scaling.py
同一思路，但拆成 x/z 两个方向的尺度，以支持薄片高纵横比）。
"""
from __future__ import annotations
import torch

from ..physics.materials import Materials


def characteristic_scales(mat: Materials, sigma0: float, L: float, H: float):
    """返回各量的特征尺度（用于输出缩放 + 损失归一化）。

    方向性尺度（L=H 时退化回单一尺度，与 baseline 一致）：
        Ux  = sigma0*L/C11            u 沿横向 x（u ~ eps_xx*L）
        Uz  = sigma0*H/C11            v 沿厚度 z（v ~ eps_zz*H）
        PHI = |d33|*sigma0*H/eps_clamped   φ 跨厚度（开路电压 ~ |g33|*σ*H）
    """
    Ux = sigma0 * L / mat.C11
    Uz = sigma0 * H / mat.C11
    PHI = abs(mat.d33) * sigma0 * H / mat.eps_clamped   # |g33| * sigma0 * H
    SIGMA = sigma0
    D = abs(mat.d33) * sigma0
    return {"Ux": Ux, "Uz": Uz, "PHI": PHI, "SIGMA": SIGMA, "D": D, "L": L, "H": H}


def top_traction(x, sigma0, load=None):
    """顶面牵引力幅值（正=压）。x: (N,1) tensor，单位 m。

    load=None                      -> 均匀 sigma0（原 baseline）
    load={"x_center","sigma","F0"} -> 高斯 F0*exp(-(x-xc)^2/(2σ^2))，单位 m/Pa
    F0 建议取 ~sigma0（峰值与参考应力同量级，损失归一化 O(1) 不变）。
    """
    if load is None:
        return torch.full_like(x, sigma0)
    xc, sig, F0 = load["x_center"], load["sigma"], load["F0"]
    return F0 * torch.exp(-((x - xc) ** 2) / (2.0 * sig ** 2))


def _first_grads(model, xy):
    """对网络输出 (u,v,phi) 求一阶导。返回 out, ux,uz, vx,vz, px,pz。"""
    out = model(xy)
    gs = [torch.autograd.grad(out[:, i].sum(), xy, create_graph=True)[0]
          for i in range(out.shape[1])]
    ux, uz = gs[0][:, 0:1], gs[0][:, 1:2]
    vx, vz = gs[1][:, 0:1], gs[1][:, 1:2]
    px, pz = gs[2][:, 0:1], gs[2][:, 1:2]
    return out, ux, uz, vx, vz, px, pz


def _constitutive(ux, uz, vx, vz, px, pz, mat: Materials):
    sxx = mat.C11 * ux + mat.C12 * vz + mat.e31 * pz
    szz = mat.C12 * ux + mat.C11 * vz + mat.e33 * pz
    txz = mat.G * (uz + vx)
    Dx = -mat.eps_clamped * px
    Dz = mat.e31 * ux + mat.e33 * vz - mat.eps_clamped * pz
    return sxx, szz, txz, Dx, Dz


def _d(f, xy, which):
    """f 对 xy 的第 which 个坐标求偏导（二阶自动微分）。"""
    return torch.autograd.grad(f.sum(), xy, create_graph=True)[0][:, which:which + 1]


def pde_residuals(model, xy, mat: Materials):
    """返回 (res1, res2, res3) = (div sigma)_x, (div sigma)_z, div D。"""
    out, ux, uz, vx, vz, px, pz = _first_grads(model, xy)
    sxx, szz, txz, Dx, Dz = _constitutive(ux, uz, vx, vz, px, pz, mat)
    res1 = _d(sxx, xy, 0) + _d(txz, xy, 1)   # sigma_xx,x + tau_xz,z
    res2 = _d(txz, xy, 0) + _d(szz, xy, 1)   # tau_xz,x + sigma_zz,z
    res3 = _d(Dx, xy, 0) + _d(Dz, xy, 1)     # D_x,x + D_z,z
    return res1, res2, res3


def fields_at(model, xy, mat: Materials):
    """在给定点求 (u, v, phi, sxx, szz, txz, Dx, Dz)。"""
    out, ux, uz, vx, vz, px, pz = _first_grads(model, xy)
    sxx, szz, txz, Dx, Dz = _constitutive(ux, uz, vx, vz, px, pz, mat)
    return (out[:, 0:1], out[:, 1:2], out[:, 2:3], sxx, szz, txz, Dx, Dz)


def loss(model, xy_int, bc_pts, mat: Materials, sigma0, L, H, weights=None, load=None):
    """总损失 = pde 残差 + 边界条件（都除以特征尺度）。

    xy_int: (n,2) numpy 数组，内部配点。
    bc_pts: dict，键 left/right/top/bottom，各 (n,2) numpy 数组。
    """
    w = {"pde": 1.0, "bc": 1.0}
    if weights is not None:
        w.update(weights)
    sc = characteristic_scales(mat, sigma0, L, H)
    dev = next(model.parameters()).device

    def t(a):
        return torch.tensor(a, dtype=torch.float32, device=dev, requires_grad=True)

    # ---- 内部 PDE 残差 ----
    # 薄片内 ∂/∂z ~ 1/H 是主导导数方向，故残差用 H 归一化（L=H 时与 baseline 一致）
    xy = t(xy_int)
    res1, res2, res3 = pde_residuals(model, xy, mat)
    loss_pde = (torch.mean((res1 * sc["H"] / sc["SIGMA"]) ** 2)
                + torch.mean((res2 * sc["H"] / sc["SIGMA"]) ** 2)
                + torch.mean((res3 * sc["H"] / sc["D"]) ** 2))

    # ---- 边界条件 ----
    loss_bc = 0.0
    # 左边界：u=0（固定），Dx=0（电绝缘）
    u, v, phi, sxx, szz, txz, Dx, Dz = fields_at(model, t(bc_pts["left"]), mat)
    loss_bc += torch.mean((u / sc["Ux"]) ** 2) + torch.mean((Dx / sc["D"]) ** 2)
    # 右边界：sxx=0, txz=0（自由），Dx=0（电绝缘）
    u, v, phi, sxx, szz, txz, Dx, Dz = fields_at(model, t(bc_pts["right"]), mat)
    loss_bc += (torch.mean((sxx / sc["SIGMA"]) ** 2)
                + torch.mean((txz / sc["SIGMA"]) ** 2)
                + torch.mean((Dx / sc["D"]) ** 2))
    # 底边界：v=0, phi=0（接地电极）
    u, v, phi, sxx, szz, txz, Dx, Dz = fields_at(model, t(bc_pts["bottom"]), mat)
    loss_bc += torch.mean((v / sc["Uz"]) ** 2) + torch.mean((phi / sc["PHI"]) ** 2)
    # 顶边界：szz=-traction(x), txz=0（施加载荷），Dz=0（开路电极）
    top = t(bc_pts["top"])
    u, v, phi, sxx, szz, txz, Dx, Dz = fields_at(model, top, mat)
    traction = top_traction(top[:, 0:1], sigma0, load)
    loss_bc += (torch.mean(((szz + traction) / sc["SIGMA"]) ** 2)
                + torch.mean((txz / sc["SIGMA"]) ** 2)
                + torch.mean((Dz / sc["D"]) ** 2))

    total = w["pde"] * loss_pde + w["bc"] * loss_bc
    return total, {"pde": float(loss_pde.detach().cpu()),
                   "bc": float(loss_bc.detach().cpu())}
