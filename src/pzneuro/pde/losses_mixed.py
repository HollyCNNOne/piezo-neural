"""PINN 混合格式损失：8 场一阶系统残差 + 边界条件。

一阶系统（8 未知 8 方程）：
    本构(应力, 3)：  sxx/szz/txz 与 C ε - e^T E 一致
    本构(电位移, 2)： Dx/Dz 与 e ε + eps E 一致
    平衡(2)：        div(sigma) = 0
    高斯(1)：        div(D) = 0

所有残差只含一阶导（平衡/高斯对主变量 σ/D 求一阶导），不含二阶导，
从根上绕开 3 场原变量格式在高纵横比下的二阶导数放大/病态问题。
"""
from __future__ import annotations
import torch

from ..physics.materials import Materials
from .losses import characteristic_scales, top_traction


def _jacobian(model, xy):
    """out (N,8)；grads[k] = (df_k/dx, df_k/dz)，各 (N,2)。"""
    out = model(xy)
    grads = [torch.autograd.grad(out[:, i].sum(), xy, create_graph=True)[0]
             for i in range(out.shape[1])]
    return out, grads


def mixed_residuals(model, xy, mat: Materials):
    """8 条一阶残差。场顺序 0=u 1=v 2=phi 3=sxx 4=szz 5=txz 6=Dx 7=Dz。"""
    out, g = _jacobian(model, xy)
    ux, uz = g[0][:, 0:1], g[0][:, 1:2]
    vx, vz = g[1][:, 0:1], g[1][:, 1:2]
    px, pz = g[2][:, 0:1], g[2][:, 1:2]
    sxx, szz, txz = out[:, 3:4], out[:, 4:5], out[:, 5:6]
    Dx, Dz = out[:, 6:7], out[:, 7:8]
    sxx_x = g[3][:, 0:1]
    szz_z = g[4][:, 1:2]
    txz_x, txz_z = g[5][:, 0:1], g[5][:, 1:2]
    Dx_x = g[6][:, 0:1]
    Dz_z = g[7][:, 1:2]

    # 本构（应力/电位移）：主变量应等于由 (u,v,phi) 一阶导算出的值
    r_sxx = sxx - (mat.C11 * ux + mat.C12 * vz + mat.e31 * pz)
    r_szz = szz - (mat.C12 * ux + mat.C11 * vz + mat.e33 * pz)
    r_txz = txz - mat.G * (uz + vx)
    r_Dx = Dx + mat.eps_clamped * px
    r_Dz = Dz - (mat.e31 * ux + mat.e33 * vz - mat.eps_clamped * pz)
    # 平衡 + 高斯：对主变量 σ/D 求一阶导
    r_eq1 = sxx_x + txz_z
    r_eq2 = txz_x + szz_z
    r_gauss = Dx_x + Dz_z
    return [r_sxx, r_szz, r_txz, r_Dx, r_Dz, r_eq1, r_eq2, r_gauss]


def loss_mixed(model, xy_int, bc_pts, mat: Materials, sigma0, L, H, weights=None, load=None):
    """总损失 = 8 条一阶残差 + 边界条件（都除以特征尺度）。"""
    w = {"pde": 1.0, "bc": 1.0}
    if weights is not None:
        w.update(weights)
    sc = characteristic_scales(mat, sigma0, L, H)
    SIGMA, D = sc["SIGMA"], sc["D"]
    Ux, Uz, PHI = sc["Ux"], sc["Uz"], sc["PHI"]
    dev = next(model.parameters()).device

    def t(a):
        return torch.tensor(a, dtype=torch.float32, device=dev, requires_grad=True)

    # ---- 内部一阶残差 ----
    xy = t(xy_int)
    r = mixed_residuals(model, xy, mat)
    loss_pde = (torch.mean((r[0] / SIGMA) ** 2) + torch.mean((r[1] / SIGMA) ** 2)
                + torch.mean((r[2] / SIGMA) ** 2)
                + torch.mean((r[3] / D) ** 2) + torch.mean((r[4] / D) ** 2)
                + torch.mean((r[5] * H / SIGMA) ** 2)
                + torch.mean((r[6] * H / SIGMA) ** 2)
                + torch.mean((r[7] * H / D) ** 2))

    # ---- 边界条件（都作用在主变量上，无需对 xy 求导） ----
    def t_bc(a):
        return torch.tensor(a, dtype=torch.float32, device=dev)

    def fields(xy_pts):
        o = model(xy_pts)
        return (o[:, 0:1], o[:, 1:2], o[:, 2:3], o[:, 3:4],
                o[:, 4:5], o[:, 5:6], o[:, 6:7], o[:, 7:8])

    loss_bc = 0.0
    # 左：u=0（固定），Dx=0（电绝缘）
    u, v, phi, sxx, szz, txz, Dx, Dz = fields(t_bc(bc_pts["left"]))
    loss_bc += torch.mean((u / Ux) ** 2) + torch.mean((Dx / D) ** 2)
    # 右：sxx=0, txz=0（自由），Dx=0（电绝缘）
    u, v, phi, sxx, szz, txz, Dx, Dz = fields(t_bc(bc_pts["right"]))
    loss_bc += (torch.mean((sxx / SIGMA) ** 2) + torch.mean((txz / SIGMA) ** 2)
                + torch.mean((Dx / D) ** 2))
    # 底：v=0, phi=0（接地电极）
    u, v, phi, sxx, szz, txz, Dx, Dz = fields(t_bc(bc_pts["bottom"]))
    loss_bc += torch.mean((v / Uz) ** 2) + torch.mean((phi / PHI) ** 2)
    # 顶：szz=-traction(x), txz=0（施加载荷），Dz=0（开路电极）
    top = t_bc(bc_pts["top"])
    u, v, phi, sxx, szz, txz, Dx, Dz = fields(top)
    traction = top_traction(top[:, 0:1], sigma0, load)
    loss_bc += (torch.mean(((szz + traction) / SIGMA) ** 2)
                + torch.mean((txz / SIGMA) ** 2) + torch.mean((Dz / D) ** 2))

    total = w["pde"] * loss_pde + w["bc"] * loss_bc
    return total, {"pde": float(loss_pde.detach().cpu()),
                   "bc": float(loss_bc.detach().cpu())}
