"""MRG 双缆膜 + 有髓 cable 自检：节点本征静息、双缆纤维静息(-80)、局部 φe 放电。

用法（服务器 pzneuro env）：
    cd /root/autodl-tmp/pzneuro
    python -u tests/check_mrg.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from pzneuro.neural.membrane_mrg import (mrg_ion_current, mrg_steady_state,
                                         rest_potential, V_REST_MRG)
from pzneuro.neural.cable import solve_cable

# 1. 节点本征静息（单隔室，无结间耦合）—— 诊断值，应比 -80 去极化（~-60）
v_node = rest_potential()
m, h, p, s = mrg_steady_state(np.array([v_node]))
i_rest = mrg_ion_current(np.array([v_node]), m, h, p, s, np.array([True]))
print("节点本征静息 = %.1f mV（单节点，~-60，靠结间轴膜漏电钉到 -80）" % v_node)
print("  该点总电流 = %.3f µA/cm² (应≈0)" % i_rest[0])
assert abs(i_rest[0]) < 1.0, "节点本征静息不是固定点"

# 2. 双缆纤维静息稳定（无外力，应停在 ~-80 不漂移）—— 升级关键验证
x = np.linspace(0, 0.2, 101)
phi_e = np.zeros_like(x)
r = solve_cable(x, phi_e, fiber_type="mrg", a_um=0.5, t_span=(0.0, 10.0),
                dt_eval=0.05, rtol=1e-4, atol=1e-4)
Vax0, Vax1 = r["V"][:, 0].mean(), r["V"][:, -1].mean()
print("双缆 MRG 无外力: 轴膜电位初值 %.1f 终值 %.1f mV（目标静息 %.0f）"
      % (Vax0, Vax1, V_REST_MRG))
print("隔室数=%d, 节点数=%d" % (r["x"].size, int(r["is_node"].sum())))
assert abs(Vax1 - V_REST_MRG) < 3.0, "双缆纤维静息未回到 -80"

# 3. 局部高斯 φe 放电（峰值 100 mV 落在节点 x=0.1 cm 上，应放电）
x = np.linspace(0, 0.2, 101)
phi_g = 100.0 * np.exp(-((x - 0.1) ** 2) / (2.0 * 0.02 ** 2))
r2 = solve_cable(x, phi_g, fiber_type="mrg", a_um=0.5,
                 t_span=(0.0, 10.0), dt_eval=0.02, rtol=1e-4, atol=1e-4)
print("局部 φe 峰值 100 mV: V_max=%.1f mV, 放电=%s"
      % (r2["V"].max(), bool(r2["V"].max() > 0.0)))
assert r2["V"].max() > 0.0, "双缆 MRG 未能放电"

print("\nMRG 双缆自检 PASS")
