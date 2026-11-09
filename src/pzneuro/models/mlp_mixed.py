"""MLP 混合格式：输入 (x,z)，输出 8 场 (u,v,phi,sxx,szz,txz,Dx,Dz)。

顺序约定：前 3 通道是 u,v,phi（与 3 场版一致），后 5 通道是应力/电位移，
这样 evaluate() 读通道 0,1,2 就能拿到 u,v,phi，两版共用同一套评估代码。

优势：把 σ/D 也当主变量，损失只需一阶导，绕开原变量格式的二阶导放大。
"""
from __future__ import annotations
import torch
from torch import nn


class MLPMixed(nn.Module):
    def __init__(self, hidden=64, layers=4,
                 u_scale=1.0, v_scale=1.0, phi_scale=1.0,
                 sigma_scale=1.0, d_scale=1.0, L=1.0, H=1.0, act=nn.Tanh):
        super().__init__()
        dims = [2] + [hidden] * layers + [8]
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                blocks.append(act())
        self.net = nn.Sequential(*blocks)
        # 8 通道各自尺度：u v phi | sxx szz txz | Dx Dz
        self.register_buffer("scales", torch.tensor([
            u_scale, v_scale, phi_scale,
            sigma_scale, sigma_scale, sigma_scale,
            d_scale, d_scale,
        ]))
        self.L = L
        self.H = H

    def forward(self, xy):
        xi = 2.0 * xy[:, 0:1] / self.L - 1.0
        eta = 2.0 * xy[:, 1:2] / self.H - 1.0
        o = self.net(torch.cat([xi, eta], dim=1))  # (N, 8) O(1)
        return o * self.scales
