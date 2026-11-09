# piezo-neural — 压电微环境「力 → 电 → 神经放电」多尺度耦合模型

> Piezoelectric microenvironment force→electricity→neural multiscale coupling.
> 面向神经修复的压电微环境建模：PVDF 压电薄膜受力产生的电场，能否以多大强度兴奋神经纤维。

```text
力（局部压痕）
   │  PINN 压电正演（8 场混合 / LBFGS）
   ▼
局部电势 φe（开路 ~39 V）
   │  物理增益 gain = 2π·f_stim·C_film·R_spread ≈ 8.9e-4（体导体 + 容性源 AC 耦合）
   ▼
轴突处细胞外电位（峰值 ~34.7 mV）
   │  激活函数 f = d²φe/dx² 作 cable 源项（Rattay 1986）
   ▼
神经放电：无髓 HH（1952） / 有髓双缆 MRG（McIntyre 2002）
```

## 核心问题

压电刺激的实际兴奋阈值是多少？当前 baseline 的诚实结论是：

| 纤维类型 | 放电阈值（轴突处 φe） | 相对物理 gain |
|---|---|---|
| HH 无髓（标准 a=1.7µm） | 35.2 mV | 1.01×（擦边不放电） |
| MRG 有髓（标准 a=1.7µm） | 85.2 mV（下界） | 2.5×（不放电） |
| 物理 gain 能提供 | 34.7 mV | — |

即：按现有物理参数（200 kHz、间隙 50 µm、组织电导 0.3 S/m），压电电场**连无髓纤维都只是勉强擦边，兴奋不了有髓纤维**。这个「有髓阈值 gap」是本项目的核心张力点，也是后续升级要闭合的目标。

## 目录

```text
src/pzneuro/
  physics/     纯 numpy：材料参数(PVDF) + 本构 + 解析解（本机可单测）
  geometry/    几何：方形域
  models/      网络：MLP / 8 场混合输出 mlp_mixed
  pde/         PINN：本构/平衡/高斯残差 + 训练循环 + 评估（train / losses / losses_mixed）
  neural/      cable 方程 + HH / MRG 膜（membrane_hh / membrane_mrg / cable）
  coupling/    接口层：激活函数 f=d²φe/dx² + 物理增益 gain + 电势取值
tests/         自检（check_physics / check_cable / check_mrg）
run_*.py       各 Rung 的端到端入口
```

## 环境

```bash
pip install -r requirements.txt   # torch>=2.0 numpy scipy matplotlib
```

- PINN 训练段需要 **torch**（服务器）；cable/HH/MRG 段纯 numpy+scipy（本机可跑）。
- 项目实际在 AutoDL 服务器 `pzneuro` 环境上跑。

## 怎么跑

各 Rung 的入口脚本对应「一步一步升级」的每一步（每步 = 一个消融实验）：

| 脚本 | 作用 |
|---|---|
| `run_baseline.py` | Rung 0–2：力→电 PINN 正演（3 场 / 8 场混合 / LBFGS） |
| `run_rung3b_pipeline.py` | Rung 3b：非均匀压痕 → φe → cable+HH 端到端 |
| `run_cable_only.py` | 只跑 cable 段（读 φe CSV，不重训 PINN） |
| `run_rung3c_compare.py` | Rung 3c：物理 gain 标定 + HH vs MRG 阈值对比 |
| `run_rung3d_diagnose.py` | Rung 3d 第一步：HH 阳极断开（anode break）机制验证 |
| `run_rung3d_compare.py` | Rung 3d 第二步：完整双缆 MRG + 远场延拓 + 阈值对比 |

典型流程：先 `run_rung3b_pipeline.py` 训 PINN 并保存 `outputs/rung3b_full/phi_e_line.csv`，
之后 `run_rung3d_compare.py --csv ... --a-um 1.7` 反复快速扫神经阈值。

## Baseline 度量（冻结值）

- PINN 电势误差：φ ~1.9e-2（8 场混合 + LBFGS），φ_top ~1.3%
- 物理增益：8.9e-4（开路 39 V → 轴突 34.7 mV）
- HH 阈值：35.2 mV（a=1.7µm）；MRG 阈值：85.2 mV（a=1.7µm，下界）

## 已知局限

1. **MRG 密封端反射**：有限纤维两端密封会把电流反射回来，使边缘节点比真实正瓣节点更早兴奋，
   MRG 阈值是**下界**（真值只会更高，不影响「MRG > HH > 物理 gain」的定性结论）。待用更长纤维 + 开放边界修（exp001）。
2. N_LAM=80 未按纤维直径缩放（τ_my=C/G 与 N 无关，不影响放电瞬态）。
3. 1µm 细纤维超出 MRG 标定范围（3.4–11.4µm 轴突）。
4. 图内中文缺字体（DejaVu 无 CJK，PNG 出现 tofu）。

## 定位与创新边界

- **工程集成创新**：把 PINN 压电正演、物理增益、cable+HH/MRG 三段拼成可复现的端到端链路。
- **算法/物理原创**：待在上述范式内以「升级 = 消融实验」方式推进（exp001、exp002…）。
