"""材料参数：单点事实来源 + d-form 到 e-form 的换算。

模块一实测得到的是应变-电荷形式（d-form）的量：
    d33 / d31（压电应变系数，C/N）、E（模量）、nu（泊松比）、eps_r（相对介电常数）。

PINN 残差里用的是应力-电荷形式（e-form）：
    sigma = C eps - e^T E
    D     = e eps + eps_s E

两者等价。平面应力、2D、工程剪应变约定下的换算公式：

    C11 = C22 = E/(1-nu^2),   C12 = nu*C11,   G = E/(2(1+nu))
    e31 = E*(d31 + nu*d33)/(1-nu^2)
    e33 = E*(nu*d31 + d33)/(1-nu^2)
    eps_s = eps_r*eps0 - (d31*e31 + d33*e33)     # 钳位（常应变）介电常数

与参考代码 PINNs_piezoelectricity-main/src/pinn_piezo/materials.py 的
C11/C12/pze_E 数值逐位一致（d31=2.2e-11, d33=-3e-11, E=2e9, nu=0.29）。
"""
from __future__ import annotations
from dataclasses import dataclass

EPS0 = 8.854187817620e-12  # 真空介电常数 F/m


@dataclass(frozen=True)
class Materials:
    E: float = 2.0e9       # 弹性模量 Pa
    nu: float = 0.29       # 泊松比
    d31: float = 2.2e-11   # 压电应变系数 C/N
    d33: float = -3.0e-11  # 压电应变系数 C/N（参考代码取负）
    eps_r: float = 12.0    # 相对介电常数

    # ---- e-form 弹性刚度 ----
    @property
    def C11(self) -> float:
        return self.E / (1.0 - self.nu ** 2)

    @property
    def C12(self) -> float:
        return self.nu * self.C11

    @property
    def C22(self) -> float:
        return self.C11

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    # ---- 压电应力系数（e-form） ----
    @property
    def e31(self) -> float:
        return self.E * (self.d31 + self.nu * self.d33) / (1.0 - self.nu ** 2)

    @property
    def e33(self) -> float:
        return self.E * (self.nu * self.d31 + self.d33) / (1.0 - self.nu ** 2)

    # ---- 介电常数 ----
    @property
    def eps_stress(self) -> float:    # 常应力介电常数
        return self.eps_r * EPS0

    @property
    def eps_clamped(self) -> float:   # 钳位（常应变）介电常数
        return self.eps_stress - (self.d31 * self.e31 + self.d33 * self.e33)


# 参考代码里的 PVDF 示例参数（模块一实测后替换这 5 个数即可）
PVDF = Materials()
