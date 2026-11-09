"""激活函数：细胞外电势场的二阶空间导数（cable 方程的源项）。

    f(x) = d²φe/dx²

现代用法：f(x) 不是「单独拿来看正负猜激活」的老式预测器，而是作为
cable 方程的细胞外强迫源项（见 Rattay 1986 的原始推导 + Joucla & Yvert 2012 综述）。
f>0 去极化（易放电）、f<0 超极化。
"""
from __future__ import annotations
import numpy as np


def activating_function_1d(x, phi_e):
    """一维：phi_e 是沿轴突 x 采样的细胞外电势（mV），返回 f(x)=d²φe/dx²。"""
    x = np.asarray(x, dtype=float)
    phi_e = np.asarray(phi_e, dtype=float)
    return np.gradient(np.gradient(phi_e, x), x)


def eval_phi_line(model, L, H, z_axon, nx=401, device=None):
    """从 PINN 模型在 z=z_axon 一行上评估 φ，返回 (x (cm), phi_e (mV))。

    注意单位：PINN 内部是 SI（x,z 单位 m、φ 单位 V），这里转成神经侧单位
    （x 单位 cm、φ 单位 mV）。z_axon 传 SI 单位 m。
    """
    import torch
    dev = device or next(model.parameters()).device
    x_m = np.linspace(0.0, L, nx)                 # SI: m
    xy = np.stack([x_m, np.full_like(x_m, z_axon)], axis=1)
    with torch.no_grad():
        phi_v = model(torch.tensor(xy, dtype=torch.float32, device=dev))[:, 2]
    phi_e = phi_v.cpu().numpy() * 1000.0          # V -> mV
    return x_m * 100.0, phi_e                     # m -> cm


def save_phi_line(model, L, H, z_axon, out_csv, nx=401, device=None):
    """提取 φe(x) 沿 z=z_axon 一行并存 CSV（SI 单位：x_m, phi_e_V）。"""
    import torch
    dev = device or next(model.parameters()).device
    x_m = np.linspace(0.0, L, nx)
    xy = np.stack([x_m, np.full_like(x_m, z_axon)], axis=1)
    with torch.no_grad():
        phi_v = model(torch.tensor(xy, dtype=torch.float32, device=dev))[:, 2].cpu().numpy()
    np.savetxt(out_csv, np.stack([x_m, phi_v], axis=1), delimiter=",",
               header="x_m,phi_e_V", comments="")
    return x_m, phi_v
