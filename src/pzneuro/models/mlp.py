"""MLP：输入 (x,z)，输出 (u,v,phi)。

两层缩放，让网络只跟 O(1) 的量打交道：
1. 输入归一化：x/L、z/H 映射到 [-1,1]（薄片高纵横比必须，否则坐标尺度悬殊、
   首层 tanh 梯度消失/条件数差）。
2. 输出缩放：网络内部输出 O(1) 的 (u_hat,v_hat,phi_hat)，再乘各自方向尺度
   u_scale(横向)/v_scale(厚度)/phi_scale 得到物理量（u~1e-7, v~1e-8, phi~几十 V）。
导数链式法则由 autograd 自动携带 2/L、2/H，物理导数不变。
"""
from __future__ import annotations
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=3, hidden=64, layers=4,
                 u_scale=1.0, v_scale=1.0, phi_scale=1.0,
                 L=1.0, H=1.0, act=nn.Tanh):
        super().__init__()
        dims = [in_dim] + [hidden] * layers + [out_dim]
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                blocks.append(act())
        self.net = nn.Sequential(*blocks)
        self.u_scale = u_scale
        self.v_scale = v_scale
        self.phi_scale = phi_scale
        self.L = L
        self.H = H

    def forward(self, xy):
        # 输入归一化到 [-1,1]
        xi = 2.0 * xy[:, 0:1] / self.L - 1.0
        eta = 2.0 * xy[:, 1:2] / self.H - 1.0
        o = self.net(torch.cat([xi, eta], dim=1))  # (N, 3)，O(1)
        return torch.stack([
            o[:, 0] * self.u_scale,
            o[:, 1] * self.v_scale,
            o[:, 2] * self.phi_scale,
        ], dim=1)
