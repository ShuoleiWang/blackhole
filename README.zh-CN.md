# Relativistic Black Hole Renderer

[English](./README.md) | **简体中文**

基于 **WebGPU / WebGL2** 的交互式相对论黑洞实时成像实验。根 URL 进入实时
双黑洞场景；旧 `?scene=binary-approx` URL 继续兼容，
`?scene=schwarzschild` 则进入交互式单黑洞场景。

双黑洞运动和波形读数来自固定版本的 `SXS:BBH:0001` Lev5 数值相对论数据：
A/B 视在视界坐标质心给出分离和轨道相位，CoM 修正的
`Extrapolated_N2.dir/Y_l2_m2.dat` 给出复数 `h22` 波形。鼠标或触摸输入会
改变相机，每个实际渲染帧都会按当前相机重新构造 GPU 光线。

这一来源升级只适用于**动力学层**。双黑洞像素仍由逐光线冻结双体位置的
多中心 **weak-field fast-light shader** 生成；运行时不读取 SXS 近区度规或
光线 transfer map。因此它**不是 NR 光线追踪**，不是已求解双黑洞时空的
成像，也不是合并阴影的定量准确结果。下一层交互物理目标是隔离的 strong-field
WebGPU 光追；该能力尚未实现。
当前没有任何已实现场景属于完整 NR 光传播、GRMHD 或高精度辐射转移求解器。

显式 Schwarzschild 场景会在 GPU 上反向积分过去指向零测地线，并用同一条
光路计算视界捕获、理想薄盘交点、频移和全天球银河背景透镜成像。

科学入口 `?scene=transfer-map-reference` 使用项目生成的
stationary analytic **Schwarzschild 与 Kerr** 参考图验证 transfer-map
完整链路。Kerr 产品只采用固定 `SXS:BBH:0001` 余留体自旋；度规与像素均由
项目按解析 Kerr 解生成，不读取 SXS 近区时空。两者采用固定 1024×576
相机、不含吸积盘，也**不是数值相对论**。它们是校准与回归 oracle，不是
双黑洞合并渲染器。

![Schwarzschild 黑洞、薄吸积盘与引力透镜化银河背景](./docs/images/blackhole-galaxy-hero.webp)

<sub>项目在 Apple Silicon 上运行 WebGPU/Metal 的 5120×2576 实际截图，保留控制面板与后端、输出模式、帧率等运行状态。银河素材：ESO/S. Brunier；经本项目测地线追踪变形、合成并转码，原素材按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 使用；完整来源见 [`assets/SOURCES.md`](./assets/SOURCES.md)。</sub>

## 产品层级与路线

| 产品层 | 状态 | 科学边界 |
| --- | --- | --- |
| 根 URL 实时双黑洞 | 已实现 | SXS 驱动动力学与逐帧 GPU 重算，但像素仍为 weak-field fast-light |
| `?scene=schwarzschild` | 已实现 | 交互式单黑洞 Schwarzschild 测地线与理想薄盘 |
| Stationary Schwarzschild/Kerr 工作台 | 已实现 | 固定相机解析真空校准、认证交付和回归 oracle；不是 merger renderer |
| 交互式 strong-field WebGPU 双黑洞 | 规划中 | 若无更强时空证据，仍只能声明 approximate fast-light |
| 4D NR slow-light 极致离线渲染 | 规划中 | 需要 ray bundles/Jacobi；若要发光画面，还需独立来源的 GRMHD/GRRT、光谱和偏振数据 |

[`docs/rendering-modes.md`](./docs/rendering-modes.md) 详细定义两条开发路线、
允许的科学声明，以及为什么 transfer-map v1 只保持 camera-specific vacuum
escape-transfer ABI，而不是完整辐射渲染格式。

## 核心特性

- **交互式 Schwarzschild 测地线**：显式单黑洞场景使用 Störmer–Verlet 数值积分 `u'' = -u + 3u²`，而不是屏幕空间扭曲。
- **统一光路合成**：同一条光线负责黑洞捕获、多个盘面交点和天空逃逸方向，银河与恒星自然产生临界环和高阶像。
- **相对论薄盘显示**：包含 Schwarzschild 圆轨道频移、`g⁴` 总辐射强度变换、近似黑体色度、表面光学深度与肢暗化。
- **实时程序化盘面**：受盘湍流启发的有限寿命噪声场随局部 Kepler 角速度平流；它是视觉近似，不是 MHD 模拟。
- **SXS 驱动的双黑洞动力学**：根场景按需加载从
  `SXS:BBH:0001/Lev5` 派生的 2,732 个样本、约 198 KiB 的紧凑轨迹：
  真实 A/B 视界质心坐标分离与相位、CoM 修正外推 `h22`、来源事件和精确
  余留体元数据。
- **可交互时间控制**：可拖动波形时间轴、从两个播放按钮暂停/继续，并可
  开关合并阶段 `0.12×` 慢放。慢放只改变墙钟播放速度，不修改 source time
  或任何物理数据。
- **明确的渲染边界**：双黑洞仍使用原有 WGSL / GLSL weak-field
  fast-light 透镜 shader；NR 派生轨迹与真实 NR 波形并不会让像素成为 NR
  光线追踪。
- **Schwarzschild / Kerr 校准工作台**：
  `?scene=transfer-map-reference` 会先认证两个内置 1024×576 stationary
  map 之一，再交给任一后端消费。Kerr 参考在精确解析 Kerr 度规中数值积分
  可分离零测地线，并使用有限距离 BL-ZAMO、恒 Kerr 半径扁球捕获面和到
  无穷远的延拓。这些 map 是离线管线的 stationary vacuum oracle，不是
  合并帧。
- **可检查的科学诊断**：稳定 URL 可显示天空、outcome、回溯时间、频移、
  null residual 或投影误差；点击 texel 可查看解码值与原始 32-byte 记录。
- **隔离的场景架构**：scene descriptor 与 shader bundle 隔离根双黑洞、显式 Schwarzschild 和固定相机科学参考，避免不同物理假设被静默混用。
- **WebGPU 优先、WebGL2 回退**：根据浏览器实际暴露的 GPU limits、纹理尺寸和 framebuffer 能力选择路径，不按芯片型号硬编码。
- **渐进式天空资源**：仓库内置 ESO 6K 与 4K 回退；可选加载 ESA/Gaia 16000×8000 全天图。
- **能力检测与 HDR 降级**：尝试请求 Display-P3、FP16 与扩展范围输出，并检查浏览器是否保留配置；否则依次退回 P3 SDR、sRGB SDR 或 WebGL2。

## 快速开始

项目没有构建步骤，也不需要安装 JavaScript 依赖。Python 仅用于启动静态服务器。

```bash
git clone https://github.com/ShuoleiWang/blackhole.git
cd blackhole
python3 -m http.server 4173
```

打开 <http://localhost:4173>。WebGPU 需要 `localhost` 或 HTTPS 安全上下文；不支持 WebGPU 时程序会自动尝试 WebGL2。

根 URL 直接进入实时双黑洞场景，旧
<http://localhost:4173/?scene=binary-approx> 选择同一场景。
打开 <http://localhost:4173/?scene=schwarzschild> 可进入交互式单黑洞场景；
打开 <http://localhost:4173/?scene=transfer-map-reference> 可进入固定相机
Schwarzschild transfer-map 参考；追加 `&reference=kerr-remnant` 可进入
stationary Kerr 余留体参考。所有路径彼此隔离。

仓库内的 6K 银河背景可以直接运行。若希望使用约 236 MiB 的 Gaia 16K 全天图，可额外执行：

```bash
./scripts/fetch_gaia_sky.sh
```

下载脚本会从 ESA 官方地址获取原图，并在安装前校验固定 SHA-256；该大文件不会提交到 Git。

## 交互

| 操作 | 效果 |
| --- | --- |
| 鼠标拖动 / 单指拖动 | 改变观测相位与圆轨道所在平面 |
| 滚轮 / 双指缩放 | 改变观测半径 |
| 双击画面 | 重置观测视角 |
| 方向键 | 微调相位与轨道平面 |
| `0` | 令观测轨道与吸积盘共面，进入严格侧视 |
| `+` / `-` | 减小 / 增大观测半径 |
| 空格 | 暂停 / 继续模拟时间 |

“科学真色 / 哈勃调色”只改变显示映射与轻量 PSF，不改变测地线、盘面遮挡或频移。

在根双黑洞场景中，拖动与缩放控制相机。每个实际渲染帧都会构造新的相机
光线并重算 weak-field fast-light 结果，不会采样固定相机 transfer map。
时间线按钮和空格键会暂停/继续同一播放状态；range 控件可拖动 protocol
time；“合并慢放”可在
`t = -160 M` 至 `t = 70 M` 启用仅用于展示的 `0.12×` 速度。波形条是真实
的 CoM 修正 SXS `Extrapolated_N2` `h22`，最大振幅对齐到 protocol
`t = 0`。由于来源是真空双黑洞，吸积率控件会被禁用。这些交互都不会改变
weak-field fast-light 渲染模型。

Transfer-map 工作台的相机与投影固定，因此拖动、缩放、重置、运动、质量、
吸积率和时间控件均禁用。它可以切换 Schwarzschild / Kerr 参考，并显示天空、
outcome、回溯时间、频移、null residual 或投影误差。点击画面可检查一条
32-byte 光线记录；方向键移动 texel，Shift 加速，Escape 关闭检查器。曝光和
画质仍只影响显示。

## 运行参数

| URL 参数 | 用途 |
| --- | --- |
| 根 URL | 打开 SXS 驱动动力学、逐帧 weak-field fast-light GPU 重算的双黑洞场景 |
| `?scene=binary-approx` | 根双黑洞场景的旧版兼容别名 |
| `?scene=schwarzschild` | 打开交互式单黑洞 Schwarzschild 测地线与理想薄盘场景 |
| `?scene=transfer-map-reference` | 打开固定相机 stationary analytic Schwarzschild transfer-map 参考；非 NR、无吸积盘 |
| `?scene=transfer-map-reference&reference=kerr-remnant` | 打开 stationary analytic Kerr 余留体自旋参考；非 NR、无吸积盘 |
| `&diagnostic=sky\|outcome\|lookback\|frequency-shift\|null-residual\|projection-error` | 选择稳定的 transfer-map 诊断视图 |
| `?renderer=webgl` | 强制使用 WebGL2 回退路径 |
| `?hdr=0` | 关闭扩展 HDR，使用稳定的 SDR 输出 |
| `?sky=high` | 固定使用仓库内的 ESO 6K 银河背景 |
| `?sky=ultra` | 启动时阻塞尝试本地 Gaia 16K 背景 |
| `?presentation=1` | 隐藏控制面板与状态栏，适合展示和截图 |

参数可以组合，例如：

```text
http://localhost:4173/?scene=transfer-map-reference&reference=kerr-remnant&diagnostic=outcome&renderer=webgl&hdr=0
```

## 渲染管线

显式 Schwarzschild 路径：

1. 从圆轨道观测者的局部共动标架生成相机光线。
2. 做 Lorentz 变换，进入局部 Schwarzschild 静态标架。
3. 在 fragment shader 中积分零测地线并判断捕获、逃逸和盘面交叉。
4. 按从近到远的顺序累积薄盘辐射与透过率，再采样逃逸方向上的全天球背景。
5. WebGPU 在 FP16 中间目标上完成光追，再根据实际显示能力输出扩展 HDR 或 SDR；WebGL2 提供 sRGB/SDR 回退。

根双黑洞路径会按需加载并做完整性校验：一个 Phase 2 manifest 及其紧凑
sample asset。运行时线性插值 SXS A/B 视在视界质心的坐标分离与展开相位、
CoM 修正外推复数 `h22`，以及单独标注为展示代理的 topology blend，再把
分离、相位和 blend 送入两个后端现有的 binary trace shader。

该 shader 在每个实际渲染帧按当前相机重新构造光线，在每条光线内冻结双体
位置，计算双中心弱场 fast-light 偏折并过渡为球对称视觉余留体；天空、
后处理和 HDR 仍复用共享阶段。它**不会**
读取 SXS 近区时空、在 NR 度规中积分零测地线、渲染余留体自旋，或把捕获面
计算为视在/事件视界。相机也不会套用单黑洞圆轨道观测者的 Lorentz boost。

参考场景完成一条独立、失败即拒绝的链路：从源码内固定信任表选择产品；
认证 manifest 原始字节、sidecar 和全部 chunk；验证 v1 schema、32-byte
记录、坐标、outcome 与精度；将 589,824 条记录上传到 WebGPU 或 WebGL2；
运行时只取最近 texel、绝不跨捕获边界混合逃逸方向，并且只有 `escaped`
才采样天空；最后复用共享后处理和 HDR/SDR 输出。

两个产品都是 `r=40M` 观测者、40° 垂直视场、1024×576 单时刻投影。
Schwarzschild map 包含 557,772 条 escaped 和 32,052 条 captured 光线。
Kerr map 使用 `a/M = 0.686461676493`、有限距离 BL-ZAMO 和 ingoing
Cartesian Kerr-Schild manifest 坐标，包含 558,684 条 escaped 和 31,140
条 captured 光线。两者均无 unusable 记录，且都不是 NR 时空、双黑洞合并
画面、吸积盘或 GRMHD / 辐射转移结果。

这些参考会作为两条开发路线共同使用的 stationary regression oracle 保留。
实时路线继续在 GPU 上生成相机光线；未来离线路线需要 4D NR slow-light ray
bundles 与独立版本的辐射产品，而不会扩张 v1 32-byte vacuum ABI 的含义。

主要实现：

- [`src/main.js`](./src/main.js)：场景选择、相机轨道、物理参数、交互和动态画质
- [`src/shaders.js`](./src/shaders.js)：显式单黑洞场景的 WGSL / GLSL Schwarzschild 测地线、薄盘辐射、天空采样与后处理
- [`src/scenes/binary-approx-scene.js`](./src/scenes/binary-approx-scene.js)：根双黑洞场景生命周期、SXS 时间线、播放 UI 与逐帧参数
- [`src/scenes/binary-dynamics-adapter.js`](./src/scenes/binary-dynamics-adapter.js)：失败即拒绝的浏览器加载、完整性检查和确定性动力学插值
- [`src/scenes/binary-playback-clock.js`](./src/scenes/binary-playback-clock.js)：拖动、帧率无关播放、末尾停留、循环与仅展示慢放
- [`src/binary-shaders.js`](./src/binary-shaders.js)：配套 WebGPU / WebGL2 弱场双黑洞 trace shader 与场景 uniform 适配
- [`src/scenes/transfer-map-reference-scene.js`](./src/scenes/transfer-map-reference-scene.js)：固定相机参考场景生命周期与失败即拒绝加载 UI
- [`src/transfer-map-loader.js`](./src/transfer-map-loader.js)：浏览器端 manifest、sidecar、chunk、ABI、outcome 与精度验证
- [`src/transfer-map-shaders.js`](./src/transfer-map-shaders.js)：WebGPU / WebGL2 最近 texel consumer
- [`assets/transfer-maps/schwarzschild-reference-v1/manifest.json`](./assets/transfer-maps/schwarzschild-reference-v1/manifest.json)：1024×576 可渲染解析参考及 9 个哈希 chunk
- [`scripts/generate_schwarzschild_transfer_map.py`](./scripts/generate_schwarzschild_transfer_map.py)：确定性离线生成器
- [`scripts/verify_schwarzschild_transfer_map.py`](./scripts/verify_schwarzschild_transfer_map.py)：独立 stationary physics 验证器
- [`assets/transfer-maps/kerr-remnant-reference-v1/manifest.json`](./assets/transfer-maps/kerr-remnant-reference-v1/manifest.json)：1024×576 可渲染解析 Kerr 参考及哈希 chunk
- [`scripts/generate_kerr_transfer_map.py`](./scripts/generate_kerr_transfer_map.py)：带完整光线 tolerance refinement 的确定性 Kerr 生成器
- [`scripts/verify_kerr_transfer_map.py`](./scripts/verify_kerr_transfer_map.py)：有限 ZAMO 阴影、Kerr-Schild 身份与独立 fixed-step 光线验证器
- [`docs/kerr-reference.md`](./docs/kerr-reference.md)：Kerr 配置、方程、验证边界与复现说明
- [`assets/scenes/binary-sxs-bbh-0001-v2.json`](./assets/scenes/binary-sxs-bbh-0001-v2.json)：Phase 2 来源、科学状态、事件、完整性、误差、渲染边界与播放 manifest
- [`assets/scenes/binary-sxs-bbh-0001-v2.samples.json`](./assets/scenes/binary-sxs-bbh-0001-v2.samples.json)：2,732 个样本的紧凑 SXS 动力学与波形轨迹
- [`scripts/generate_binary_sxs_dynamics.py`](./scripts/generate_binary_sxs_dynamics.py)：从三个固定官方 SXS 文件离线确定性生成轨迹
- [`scripts/verify_binary_dynamics.py`](./scripts/verify_binary_dynamics.py)：失败即拒绝的 Phase 2 来源、资产、事件、插值、余留体、渲染边界与播放检查
- [`tests/binary-playback.test.mjs`](./tests/binary-playback.test.mjs)：source anchor、插值、拖动、慢放、末尾停留与帧率无关性的 Node 测试
- [`assets/scenes/binary-pn-equal-mass-v1.json`](./assets/scenes/binary-pn-equal-mass-v1.json)：仅为回归保留的 legacy v1 PN/现象学资产
- [`scripts/verify_binary_preview.py`](./scripts/verify_binary_preview.py)：legacy PN 契约与未改变弱场 shader 的收敛回归
- [`docs/binary-model.md`](./docs/binary-model.md)：双黑洞科学边界、当前弱场模型与未来实时/离线架构
- [`docs/rendering-modes.md`](./docs/rendering-modes.md)：已实现产品层级、实时
  strong-field 路线、离线 NR/GRRT 路线与科学声明等级
- [`docs/nr-transfer-map-v1.md`](./docs/nr-transfer-map-v1.md)：transfer map
  v1 协议的规范术语、字段语义、安全规则与阶段状态
- [`schemas/nr-transfer-map-v1.schema.json`](./schemas/nr-transfer-map-v1.schema.json)：机器可读的 transfer map manifest schema
- [`assets/transfer-maps/contract-fixture-v1/manifest.json`](./assets/transfer-maps/contract-fixture-v1/manifest.json)：项目生成的小型一致性 fixture；不含 NR 派生 payload
- [`scripts/generate_nr_contract_fixture.py`](./scripts/generate_nr_contract_fixture.py)：确定性重建一致性 fixture
- [`scripts/verify_nr_contract.py`](./scripts/verify_nr_contract.py)：失败即拒绝的 manifest、sidecar、坐标标架和逐光线记录验证器
- [`tests/test_nr_contract.py`](./tests/test_nr_contract.py)：协议正向与对抗性回归测试
- [`src/webgpu-renderer.js`](./src/webgpu-renderer.js)：WebGPU 双阶段渲染与 HDR/P3 配置协商
- [`src/webgl-renderer.js`](./src/webgl-renderer.js)：WebGL2 硬件回退与半浮点 framebuffer 探测

## 模型范围与限制

| 场景 / 组件 | 已实现 | 当前边界 |
| --- | --- | --- |
| 显式单黑洞 | 非旋转 Schwarzschild 时空与 GPU 零测地线数值积分 | 不支持 Kerr 自旋和 frame dragging；最窄临界曲线仍受采样限制 |
| 显式 Schwarzschild 吸积盘 | `r = 6M` 至 `18M` 的理想零厚度表面、频移、近似发射与受湍流启发的结构 | 不含有限尺度高度、GRMHD、完整光谱、偏振或自洽辐射转移 |
| 双黑洞轨道动力学 | SXS:BBH:0001 Lev5 A/B 视在视界惯性坐标质心从 relaxation 至最后一对 A/B 样本的分离与展开相位 | 是真实 NR 诊断量，但依赖坐标和规范；单体视界数据结束后只保持最后分离/相位，不虚构后续轨迹 |
| 双黑洞波形 | CoM 修正 `Extrapolated_N2` 复数 `h22`，最大振幅对齐到 protocol `t = 0` | 远区波形不是近区度规，不能决定相机光线传播 |
| 双黑洞合并/余留体数据 | 共同视在视界事件 `t = -6.072285 M`；精确元数据余留质量 `0.951609417715 M`、自旋向量 `(-7.29520687012e-10, 7.40468371215e-10, 0.686461676493)` | 从共同视界事件到波形峰值的 topology blend 只是展示代理；shader 不渲染视界几何、反冲、Kerr 自旋或 frame dragging |
| 双黑洞透镜 | 每条光线内冻结两个弱场单极子的 fast-light 偏折，随后过渡为球对称余留体代理 | 不是强场测地线积分；不渲染余留体自旋和 frame dragging，精细光子环、焦散、时延和视界拓扑不具定量可信度 |
| 双黑洞发射 | 无吸积盘的真空天空透镜 | 若加入发光等离子体，需要物理气体初始条件、GRMHD 与辐射转移 |
| Stationary Schwarzschild 参考 | 固定 1024×576 解析真空 map、认证 chunk、WebGPU/WebGL2 最近 texel playback | 固定相机；无吸积盘、NR 来源、时间插值或双黑洞 slow-light 光线 |
| Stationary Kerr 余留体参考 | 在 `a/M = 0.686461676493` 的精确解析 Kerr 度规中数值积分真空零测地线，并使用有限 BL-ZAMO、扁球 Kerr-r 捕获面、认证 playback 与诊断 | 只使用 SXS 余留体自旋参数；无 SXS 近区度规、双黑洞时间依赖、发射模型或 NR 派生像素 |
| NR transfer-map 协议 | 版本化 schema、合成 fixture、失败即拒绝验证器、参考 consumer 与回归测试 | consumer 只由解析数据验证；仓库仍无 NR 派生 transfer map |
| 共享渲染器 | WebGPU 主路径、WebGL2 回退 | HDR、P3、FP16 与 16K 纹理由运行时能力决定；HDR 不提高模型精度 |

产品层级与两条开发路线见
[`docs/rendering-modes.md`](./docs/rendering-modes.md)；Schwarzschild
几何单位、临界轨道、侧视图像和颜色不对称见
[`docs/physics-notes.md`](./docs/physics-notes.md)；stationary Kerr 产品见
[`docs/kerr-reference.md`](./docs/kerr-reference.md)；双黑洞模型物理边界与
未来离线 NR 架构见 [`docs/binary-model.md`](./docs/binary-model.md)。

## M3 Pro 兼容性与 HDR

当前硬件目标是 **M3 Pro**，已实测 WebGPU/Metal、
WebGL2/ANGLE-on-Metal 回退、Display-P3 FP16、SDR 降级和 16K 背景升级。
程序仍会在运行时协商纹理上限、canvas format、半浮点 framebuffer 与显示动态
范围；本文不再作单独的 M4 兼容承诺。

右上角状态栏显示实际后端、GPU、输出模式、FPS 与内部渲染分辨率。动态画质会在用户设置的上限内调整普通光线步数与分辨率；临界冲量参数附近的光线保持更高积分预算。

## 验证

```bash
python3 scripts/verify_physics.py
python3 scripts/verify_binary_dynamics.py
node --test tests/binary-playback.test.mjs
python3 scripts/verify_binary_preview.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_nr_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_nr_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_nr_contract.py assets/transfer-maps/schwarzschild-reference-v1/manifest.json
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_schwarzschild_transfer_map.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_schwarzschild_transfer_map.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_nr_contract.py assets/transfer-maps/kerr-remnant-reference-v1/manifest.json
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_kerr_transfer_map.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_kerr_transfer_map.py
node --test tests/transfer-map-runtime.test.mjs
```

需要重建内置参考资产时，先运行：

```bash
python3 scripts/generate_schwarzschild_transfer_map.py
python3 scripts/generate_kerr_transfer_map.py
```

Schwarzschild 数值回归覆盖：

- 临界冲量参数 `b_c = 3√3 M`
- 弱场偏折与 `4M/b` 的一致性
- 有限距离观察者的阴影角直径
- 零测地线积分守恒量
- 184 / 288 步实时预算下的捕获与逃逸行为

Phase 2 双黑洞验证器会按 URL、大小、MD5 和 SHA-256 固定三个官方来源文件；
检查 2,732-sample sidecar 的哈希和 schema；确认 SXS 事件顺序、`h22` 峰值
位于 `t = 0`、共同视界位于 `t = -6.072285 M`、精确元数据余留体参数、
严格 validity 的视界后保持策略，以及全部声明插值界。实测最大轨道相位插值
残差为 `6.442e-4 rad`。Node 测试覆盖事件 anchor、有限插值、scrub clamp、
仅展示慢放、确定性循环/末尾停留和帧率无关性。

Legacy 双黑洞回归仍检查旧 PN manifest，并用与未改变 shader 方程一致的
90×45 CPU 光线网格验证代表性宽双体、过渡与余留体视角在固定 512 步生产
预算下没有未收敛光线。两者验证的是来源完整性、动力学播放与声明的实时
收敛门槛；它们**不验证** NR 光传播、已求解双黑洞时空、强场透镜或定量
准确的合并画面。

NR 契约检查严格 JSON/schema、一致的 sidecar 哈希与大小、可移植 artifact
定位规则、连续有序 chunk、物理系统与 source/protocol 时间声明、互逆的空间
仿射标架、正确的 ICRS 旋转、观测者 tetrad 正交归一性、光线积分与边界语义，
以及合法且有限的逐光线结果。数据声明可渲染前，还会把 outcome fractions
与解码记录交叉核对。未知或缺失字段、重复键、非有限数、路径越界和含混的
无效光线状态会被拒绝。验证通过表示 **protocol-conformant**，不表示
**NR-backed** 或 **physically validated**。

Schwarzschild verifier 会独立恢复有限距离阴影直径 `14.548010°` 与边界频移
`g = 1.024951860`，并报告最大采样解析 null residual `7.678e-14`、最大独立
方向误差 `1.062e-8 rad`、最大逐光线投影估计 `1.415e-2 px`。这些是
stationary reference 检查；NR 收敛阶与约束范数字段为 `not-applicable`，
不是数值为零的 NR 测量。

Kerr verifier 会独立重建 Cartesian Kerr-Schild 度规与 BL-ZAMO tetrad，
计算有限距离 spherical-photon 临界曲线、检查完整 capture mask，并用
fixed-step RK4 重追代表性完整光线。更完整的 Kerr 验证套件另用生成器级单元
回归检查 Schwarzschild 极限与自旋翻转镜像；独立 verifier 检查第一积分分离、
无穷远尾段、null residual 与逐记录投影门槛。内置 map 的解析 capture-mask
mismatch 为 0，最大 stored null residual 为 `3.068e-9`，p95 / 最大投影估计为
`1.929e-4 / 3.752e-3 px`，最大独立方向误差为 `8.679e-9 rad`。精确模型与验收边界见
[`docs/kerr-reference.md`](./docs/kerr-reference.md)。

这些检查共同覆盖一组明确的数值性质与架构契约，但不等价于完整画面、辐射模型或所有 GPU 的自动化验证。当前仓库尚未配置 GPU 图像回归 CI。

## 天空素材与署名

- **ESA/Gaia/DPAC · A. Moitinho**：可选 16000×8000 Gaia EDR3 全天图，CC BY-SA 3.0 IGO。
- **ESO/S. Brunier**：仓库内置 6000×3000 银河摄影背景，CC BY 4.0。
- `assets/deep-field.webp`：由仓库脚本生成的备用深空素材，不是默认背景。

下载地址、处理方式、哈希和完整许可信息见 [`assets/SOURCES.md`](./assets/SOURCES.md)。第三方素材不会因本项目代码未来采用某种许可证而被重新授权。

## 许可证

当前仓库尚未声明项目代码许可证。第三方天空素材、SXS 派生数据与 vendored
依赖仍分别遵循其来源条款。Phase 2 使用的固定 Zenodo record 没有声明
license，因此本仓库只记录该事实，不虚构 SPDX 标识，也不从其他页面推断
license。在选择项目代码许可证前，请不要假设仓库内容已按 MIT、
Apache-2.0 等许可证授权。
