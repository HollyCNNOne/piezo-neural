"""pzneuro：压电微环境「力→电→神经放电」的多文件解耦实现。

目录职责：
  physics   纯 numpy：材料换算、本构、解析解
  geometry  几何插件
  models    神经网络
  pde       PINN 训练与评估
  neural    FHN 动力学
  coupling  电势 -> 刺激电流 接口
  inverse   （SCI 扩展，后续加）
"""
__version__ = "0.1.0"
