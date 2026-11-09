"""Rung 3a 自检：HH 膜动力学 + 激活函数解析校验（不依赖 torch/PINN）。

运行（在有 scipy 的环境）：
    python tests/check_cable.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from pzneuro.neural.membrane_hh import solve_single_hh
from pzneuro.neural.fhn import detect_spikes
from pzneuro.coupling.activating_function import activating_function_1d


def check_hh_fires():
    """单隔室 HH 在阈上电流下应反复放电。"""
    t, V = solve_single_hh(I_ext=10.0, t_span=(0.0, 100.0))  # 10 µA/cm² > 阈值
    n = detect_spikes(t, V, threshold=0.0)
    print(f"[HH] 100 ms 内放电 {n} 次（阈上电流 10 µA/cm²，期望 >0）")
    return n > 0


def check_activating_function():
    """高斯 φe 的 f(x)=d²φe/dx² 与解析式对比。"""
    x = np.linspace(-1.0, 1.0, 401)
    sig = 0.2
    phi0 = 50.0  # mV
    phi_e = phi0 * np.exp(-x ** 2 / (2 * sig ** 2))
    # 解析 d²φe/dx² = phi0 * (x²-sig²)/sig⁴ * exp(-x²/(2sig²))
    f_ana = phi0 * (x ** 2 - sig ** 2) / sig ** 4 * np.exp(-x ** 2 / (2 * sig ** 2))
    f_num = activating_function_1d(x, phi_e)
    err = np.max(np.abs(f_num - f_ana)) / np.max(np.abs(f_ana))
    print(f"[激活函数] 与解析二阶导最大相对误差 = {err:.2e}（期望 <1e-2）")
    return err < 1e-2


if __name__ == "__main__":
    ok = all([check_hh_fires(), check_activating_function()])
    print("Rung 3a 自检:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
