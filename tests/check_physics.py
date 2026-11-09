"""纯 numpy 物理自检（本机无需 torch 即可运行）：验证材料换算 + 解析解自洽。

用法：
    python tests/check_physics.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from pzneuro.physics.materials import PVDF
from pzneuro.physics.constitutive import stress_electric, exact_solution

m = PVDF
print("C11          = %.6e  (期望 2.1836e9)" % m.C11)
print("C12          = %.6e  (期望 6.332e8)" % m.C12)
print("e31          = %.6e  (期望 2.904e-2)" % m.e31)
print("e33          = %.6e  (期望 -5.158e-2)" % m.e33)
print("eps_clamped  = %.6e  (期望 1.041e-10)" % m.eps_clamped)

sigma0, L, H = 1.0e6, 1.0, 1.0
ex = exact_solution(m, sigma0, L, H)
eps_xx, eps_zz, phi_z = ex["eps_xx"], ex["eps_zz"], ex["phi_z"]
print("\n解析解: eps_xx=%.6e  eps_zz=%.6e  phi_z=%.6e" % (eps_xx, eps_zz, phi_z))
print("phi_top = %.6e V" % (phi_z * H))

# 把解析解代入本构，验证 sigma_xx=0, sigma_zz=-sigma0, Dz=0（平衡+高斯自洽）
sxx, szz, txz, Dx, Dz = stress_electric(eps_xx, 0.0, 0.0, eps_zz, 0.0, phi_z, m)
print("\n代入本构验证（应力/电位移自洽）：")
print("  sigma_xx = %.6e  (期望 0)" % sxx)
print("  sigma_zz = %.6e  (期望 -1e6)" % szz)
print("  tau_xz   = %.6e  (期望 0)" % txz)
print("  Dx       = %.6e  (期望 0)" % Dx)
print("  Dz       = %.6e  (期望 0)" % Dz)

assert abs(sxx) < 1e-3, "sigma_xx 不为 0"
assert abs(szz + sigma0) < 1e-3, "sigma_zz != -sigma0"
assert abs(txz) < 1e-3, "tau_xz 不为 0"
assert abs(Dx) < 1e-3, "Dx 不为 0"
assert abs(Dz) < 1e-3, "Dz 不为 0（开路条件）"
print("\nPASS：解析解满足本构 + 平衡 + 高斯，物理层自洽。")
