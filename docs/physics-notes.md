# Physics notes

本文档说明项目当前实现的 Schwarzschild 成像模型、它能表达的物理现象，以及实时近似的边界。

## Schwarzschild 几何

着色器使用几何单位：

```text
G = c = M = 1
```

因此：

- 事件视界：`r = 2M`
- 光子球：`r = 3M`
- 临界冲量参数：`b_c = 3√3 M`
- 非旋转薄盘 ISCO：`r = 6M`

每个像素从观察者相机反向追踪一条过去指向的光线。把轨迹写成 `u = 1/r` 后，Schwarzschild 零测地线满足：

```text
u'' = -u + 3u²
```

程序使用固定角步长的 Störmer–Verlet 积分，并在光线跨越视界、逃逸到无穷远或穿过盘面时做插值求交。普通光线使用动态实时预算；临界曲线附近最多使用 384 步。

## Stationary transfer-map 参考

`?scene=transfer-map-reference` 不运行上述实时 shader，而是消费离线生成的
1024×576、9-chunk Schwarzschild 解析真空 transfer map。它采用 `r=40M`
静止观测者与固定 40° 垂直视场，只按最近 texel 读取预计算逃逸方向和 outcome，
不在捕获/逃逸 separatrix 两侧混合方向。该场景不含吸积盘，也不是 NR、
双黑洞或 GRMHD 模拟；完整 ABI、ICRS 轴映射、精度指标与验证命令见
[`nr-transfer-map-v1.md`](./nr-transfer-map-v1.md)。

同一工作台还提供
`?scene=transfer-map-reference&reference=kerr-remnant`。它使用解析 stationary
Kerr 度规与 `a/M=0.686461676493`，从 `r=40M` 的有限距离
Boyer-Lindquist ZAMO 反向追踪光线。捕获面是恒 Kerr 半径的扁球 stretched
horizon，不是欧氏球；逃逸方向从 `r=1000M` 继续积分到无穷远后才写入 ICRS。
SXS 只提供固定余留体自旋参数，不提供度规或像素，因此它仍是项目生成的解析
参考，不是 NR ray tracing。方程、来源约束和独立验证见
[`kerr-reference.md`](./kerr-reference.md)。

工作台可显示天空、outcome、回溯坐标时间、频移 `g`、null residual 和投影
误差；点击 texel 会解码其原始 32-byte 光线记录。这些伪彩诊断不改变物理
数据。

## 运动观察者

相机位于经过黑洞中心的 Schwarzschild 圆轨道平面上。屏幕光线首先从观察者的局部共动标架做 Lorentz 变换，再进入局部 Schwarzschild 静态标架；观察者自身的运动因此会影响像差和频移。

拖动操作改变观测相位和圆轨道平面。按 `0` 会使整个观测轨道与吸积盘共面，形成严格侧视极限。

## 薄吸积盘近似

盘面是 `r = 6M` 到 `18M` 的理想零厚度表面。径向通量采用零力矩薄盘启发式轮廓：

```text
F(r) ∝ x³(1 - √x),  x = r_ISCO / r
```

UI 输入的是无量纲 Eddington 光度比 `L/L_Edd`；在固定效率归一化下，峰值温标按 `[(L/L_Edd) / M]^(1/4)` 缩放。显示颜色由 640、530、460 nm 三个 Planck 样本近似得到，因此它是适合实时渲染的可见色度估计，不是完整光谱积分。

程序计算 Schwarzschild 圆轨道发射体的频移因子 `g`：

- 发射温度按 `g` 移动；
- 玻尔计辐射按 `g⁴` 转移；
- 轨道速度朝向观察者的一侧（approaching side）被 Doppler 增亮并偏向暖白；
- 轨道速度远离观察者的一侧（receding side）更暗、更偏橙红。

所以盘面左右亮度不对称是模型预期，而不是需要消除的色差。

表面光学深度、覆盖率和肢暗化属于实时薄层近似。程序会累积同一光线的多次盘面交叉，但不会沿测地线积分有限厚度三维等离子体的连续发射与吸收。

## 程序化湍流

盘面纹理使用随局部 Kepler 角速度平流的有限寿命噪声场，并在相邻噪声代之间交叉淡化。它用于避免固定旋臂和静态贴图感，只应称为“受盘湍流启发的程序化结构”。

该实现没有求解磁流体方程，不能替代 GRMHD/MRI 数值模拟，也不应被用于推断真实等离子体统计量。

## 为什么侧视图会形成环和细线

理想零厚度盘在严格侧视时，其直接投影退化为零测度的线；逐像素中心采样可能完全漏掉这条直接像。黑洞附近的完整图像仍不会消失：远侧盘发出的光会被强引力弯折，在阴影上下形成次级像，并在临界曲线附近形成更窄的高阶像。

真实吸积流具有有限尺度高度 `H(R)`。要得到严格侧视下有物理厚度的窄带，需要加入三维发射与吸收积分，而不是在屏幕空间简单加粗当前二维盘面。

## 数值边界

- 显式 `?scene=schwarzschild` 实时单黑洞模型固定为非旋转
  Schwarzschild 时空，不包含 Kerr 自旋与 frame dragging；独立的离线 Kerr
  校准参考不会改变该场景的 shader。
- 临界光子轨道附近的高阶像会指数级变窄，最终受步数上限和像素覆盖限制。
- 临界曲线窄带使用固定 2×2 测地线覆盖采样；其他区域通常只追踪像素中心光线。
- 盘面发射、三波段色度、光学深度和湍流都是面向实时显示的近似。
- HDR 是显示输出能力，不会提高测地线或辐射模型本身的物理精度。

## References

- J.-P. Luminet, *Image of a spherical black hole with thin accretion disk*, Astronomy & Astrophysics 75, 228–235 (1979). [ADS](https://ui.adsabs.harvard.edu/abs/1979A%26A....75..228L/abstract)
- O. James et al., *Gravitational lensing by spinning black holes in astrophysics, and in the movie Interstellar*, Classical and Quantum Gravity 32, 065001 (2015). [DOI](https://doi.org/10.1088/0264-9381/32/6/065001)
