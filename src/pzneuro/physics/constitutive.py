"""本构（e-form）的 numpy 参考实现 + 解析精确解。

这一层不碰 torch。PINN 的损失里用自动微分的同款公式，这里用 numpy 再实现一份，
用于 (1) 生成精确解、(2) 校验、(3) 单测。

约定：x 为水平方向，z 为厚度/极化方向。工程剪应变 gamma_xz = u_z + v_x。
E = -grad(phi)，即 Ex = -phi_x, Ez = -phi_z。
"""
from __future__ import annotations
import numpy as np

from .materials import Materials


def stress_electric(ux, uz, vx, vz, px, pz, mat: Materials):
    """由一阶导数算应力/电位移。

    参数：ux,uz,vx,vz 是位移梯度；px=phi_x, pz=phi_z。
    返回 (sxx, szz, txz, Dx, Dz)。
    """
    sxx = mat.C11 * ux + mat.C12 * vz + mat.e31 * pz   # - e31*Ez = + e31*pz
    szz = mat.C12 * ux + mat.C11 * vz + mat.e33 * pz
    txz = mat.G * (uz + vx)
    Dx = -mat.eps_clamped * px
    Dz = mat.e31 * ux + mat.e33 * vz - mat.eps_clamped * pz
    return sxx, szz, txz, Dx, Dz


def exact_solution(mat: Materials, sigma0: float, L: float = 1.0, H: float = 1.0):
    """均匀轴向压缩 + 开路 的解析解。

    载荷：sigma_zz = -sigma0（压缩）、sigma_xx = 0、tau_xz = 0、Dz = 0（开路）。
    未知 (eps_xx, eps_zz, phi_z) 满足 3x3 线性系统：

        [0]        [C11  C12   e31 ][eps_xx]
        [-sigma0] =[C12  C11   e33 ][eps_zz]
        [0]        [e31  e33  -eps_s][phi_z ]

    解得常数后积分（取左下角为 0 参考点）：
        u = eps_xx*x,  v = eps_zz*z,  phi = phi_z*z

    返回 dict(eps_xx, eps_zz, phi_z, u, v, phi)，u/v/phi 是 (x,z)->float 函数。
    """
    A = np.array([
        [mat.C11, mat.C12, mat.e31],
        [mat.C12, mat.C11, mat.e33],
        [mat.e31, mat.e33, -mat.eps_clamped],
    ])
    b = np.array([0.0, -sigma0, 0.0])
    eps_xx, eps_zz, phi_z = np.linalg.solve(A, b)

    def u(x, z):
        return eps_xx * x

    def v(x, z):
        return eps_zz * z

    def phi(x, z):
        return phi_z * z

    return {"eps_xx": eps_xx, "eps_zz": eps_zz, "phi_z": phi_z,
            "u": u, "v": v, "phi": phi}
