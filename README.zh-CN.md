# Relativistic Black Hole Renderer

[English](./README.md) | **简体中文**

基于 **WebGPU / WebGL2** 的交互式相对论黑洞实时成像实验。

默认场景在 Schwarzschild 时空中反向积分过去指向的零测地线，并用同一条光路计算事件视界捕获、理想薄吸积盘交点、相对论频移，以及全天球银河背景的引力透镜成像。

项目另有完全隔离、需要显式选择的 `?scene=binary-approx`
等质量无自旋双黑洞预览。该模式把最低阶后牛顿（PN）螺旋靠近时间线、
现象学合并/余留体过渡和实时多中心弱场光线偏折组合起来。它是
**PN／现象学弱场预览**：只有取整后的余留体参考值来自
`SXS:BBH:0001`，不使用 SXS 波形、视界、时空或光线传输数据。项目面向
实时可视化与教学演示，不是 Kerr、完整 NR、GRMHD 或高精度辐射转移求解器。

![Schwarzschild 黑洞、薄吸积盘与引力透镜化银河背景](./docs/images/blackhole-galaxy-hero.webp)

<sub>项目在 Apple Silicon 上运行 WebGPU/Metal 的 5120×2576 实际截图，保留控制面板与后端、输出模式、帧率等运行状态。银河素材：ESO/S. Brunier；经本项目测地线追踪变形、合成并转码，原素材按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 使用；完整来源见 [`assets/SOURCES.md`](./assets/SOURCES.md)。</sub>

## 核心特性

- **逐像素零测地线积分**：使用 Störmer–Verlet 数值积分 `u'' = -u + 3u²`，而不是屏幕空间扭曲。
- **统一光路合成**：同一条光线负责黑洞捕获、多个盘面交点和天空逃逸方向，银河与恒星自然产生临界环和高阶像。
- **相对论薄盘显示**：包含 Schwarzschild 圆轨道频移、`g⁴` 总辐射强度变换、近似黑体色度、表面光学深度与肢暗化。
- **实时程序化盘面**：受盘湍流启发的有限寿命噪声场随局部 Kepler 角速度平流；它是视觉近似，不是 MHD 模拟。
- **可选双黑洞预览**：`?scene=binary-approx` 按需加载独立 PN 时间线及配套 WGSL / GLSL 弱场透镜 shader。真空场景刻意不加入发光吸积盘，也不声称强场或合并阶段具有定量精度。
- **版本化 NR transfer map 边界**：失败即拒绝的 schema、确定性一致性
  fixture、生成器、验证器和测试定义了未来相机特定 slow-light
  数据进入项目的方式。它目前只是数据契约，不包含 NR 派生 transfer map
  或 NR 回放场景。
- **不破坏原功能的场景架构**：可选 scene descriptor 与 shader bundle 扩展共享 WebGPU / WebGL2 后端；默认 URL 仍使用原有 Schwarzschild shader、观测者模型、吸积盘和交互。
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

通过界面中的场景选择器，或直接打开
<http://localhost:4173/?scene=binary-approx>，可以进入实验性双黑洞预览；回到默认 URL 即恢复 Schwarzschild 场景。
transfer map 契约不会增加新的可运行场景；默认 Schwarzschild 路径和显式
选择的 `binary-approx` 路径均未改变。

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

在双黑洞预览中，拖动与缩放仍控制相机，空格暂停时间线；由于模型是真空双黑洞，吸积率控件会被禁用。波形条只是最低阶/现象学的紧凑展示，不是数值相对论波形产品。

## 运行参数

| URL 参数 | 用途 |
| --- | --- |
| `?scene=binary-approx` | 显式进入隔离的 PN／现象学弱场双黑洞预览；默认仍为 Schwarzschild |
| `?renderer=webgl` | 强制使用 WebGL2 回退路径 |
| `?hdr=0` | 关闭扩展 HDR，使用稳定的 SDR 输出 |
| `?sky=high` | 固定使用仓库内的 ESO 6K 银河背景 |
| `?sky=ultra` | 启动时阻塞尝试本地 Gaia 16K 背景 |
| `?presentation=1` | 隐藏控制面板与状态栏，适合展示和截图 |

参数可以组合，例如：

```text
http://localhost:4173/?scene=binary-approx&presentation=1&sky=high&hdr=0
```

## 渲染管线

默认 Schwarzschild 路径：

1. 从圆轨道观测者的局部共动标架生成相机光线。
2. 做 Lorentz 变换，进入局部 Schwarzschild 静态标架。
3. 在 fragment shader 中积分零测地线并判断捕获、逃逸和盘面交叉。
4. 按从近到远的顺序累积薄盘辐射与透过率，再采样逃逸方向上的全天球背景。
5. WebGPU 在 FP16 中间目标上完成光追，再根据实际显示能力输出扩展 HDR 或 SDR；WebGL2 提供 sRGB/SDR 回退。

可选双黑洞路径会按需加载并校验版本化场景清单，插值 PN/现象学时间线，再向两个渲染后端提供专用 trace shader。该 shader 在每条光线内冻结当前时间样本，使用 fast-light 的双中心弱场偏折，并逐步混合为单一视觉余留体；天空采样、后处理和 HDR 仍复用共享阶段。相机不会套用单黑洞圆轨道观测者的 Lorentz boost。捕获面与共同余留体过渡都不是计算得到的事件视界。

与运行路径分离的是一套版本化 NR transfer map **接入契约**，用于未来离线
生成的数据。当前没有运行时 consumer，也不改变两个 renderer。manifest
通过契约验证只表示 protocol-conformant，不能据此判断 NR 时空、零测地线解
或最终画面在物理上正确。

主要实现：

- [`src/main.js`](./src/main.js)：场景选择、相机轨道、物理参数、交互和动态画质
- [`src/shaders.js`](./src/shaders.js)：默认 WGSL / GLSL Schwarzschild 测地线、薄盘辐射、天空采样与后处理
- [`src/scenes/binary-approx-scene.js`](./src/scenes/binary-approx-scene.js)：可选场景生命周期、清单插值、双黑洞 UI 与逐帧参数
- [`src/binary-shaders.js`](./src/binary-shaders.js)：配套 WebGPU / WebGL2 弱场双黑洞 trace shader 与场景 uniform 适配
- [`assets/scenes/binary-pn-equal-mass-v1.json`](./assets/scenes/binary-pn-equal-mass-v1.json)：版本化、机器可读的 PN/现象学时间线及精度契约
- [`scripts/verify_binary_preview.py`](./scripts/verify_binary_preview.py)：清单 schema、对称性、PN 方程、单调性与参数边界检查
- [`docs/binary-model.md`](./docs/binary-model.md)：双黑洞科学边界、协议状态与离线 NR transfer map 架构
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
| 默认单黑洞 | 非旋转 Schwarzschild 时空与 GPU 零测地线数值积分 | 不支持 Kerr 自旋和 frame dragging；最窄临界曲线仍受采样限制 |
| 默认吸积盘 | `r = 6M` 至 `18M` 的理想零厚度表面、频移、近似发射与受湍流启发的结构 | 不含有限尺度高度、GRMHD、完整光谱、偏振或自洽辐射转移 |
| 双黑洞螺旋 | 等质量、无自旋的最低阶圆轨道 PN 样本 | 只适用于弱场绝热阶段；不含高阶 PN/EOB 演化或已求解的近区度规 |
| 双黑洞合并/余留体 | 带代表性余留体参数的现象学过渡 | 不是 Einstein 方程解；没有计算共同视界、反冲或物理 ringdown 振幅 |
| 双黑洞透镜 | 每条光线内冻结两个弱场单极子的 fast-light 偏折，随后过渡为球对称余留体代理 | 不是强场测地线积分；不渲染余留体自旋和 frame dragging，精细光子环、焦散、时延和视界拓扑不具定量可信度 |
| 双黑洞发射 | 无吸积盘的真空天空透镜 | 若加入发光等离子体，需要物理气体初始条件、GRMHD 与辐射转移 |
| NR transfer map 协议 | 版本化 schema、确定性合成 fixture、失败即拒绝的验证器与回归测试 | 接口达到 contract-conformant，可接入未来 NR 派生数据；尚无 NR 度规、transfer map 或回放场景 |
| 共享渲染器 | WebGPU 主路径、WebGL2 回退 | HDR、P3、FP16 与 16K 纹理由运行时能力决定；HDR 不提高模型精度 |

Schwarzschild 几何单位、临界轨道、侧视图像和颜色不对称见 [`docs/physics-notes.md`](./docs/physics-notes.md)；双黑洞预览的物理边界、权威参考、已实现数据契约与离线 NR → transfer map 架构见 [`docs/binary-model.md`](./docs/binary-model.md)。

## 兼容性与 HDR

渲染器没有 M3、M4 或其他 GPU 型号的专用分支。它依据浏览器返回的 texture limits、canvas 配置、半浮点 framebuffer 完整性以及显示动态范围逐级选择能力，因此同一代码可以在不同 Apple Silicon 上使用相应的 WebGPU/Metal 或 WebGL2/Metal 路径。

- **M3 Pro**：已实测 WebGPU/Metal、WebGL2/Metal、Display-P3 FP16 路径、SDR 降级和 16K 后台升级。
- **M4**：设计上使用相同的能力协商路径，不依赖 M4 独有功能；当前仓库尚未记录 M4 实机 smoke test。
- **其他平台**：能否启用 WebGPU、HDR 或大纹理由浏览器、操作系统、驱动、显示器及窗口所在屏幕共同决定。

右上角状态栏显示实际后端、GPU、输出模式、FPS 与内部渲染分辨率。动态画质会在用户设置的上限内调整普通光线步数与分辨率；临界冲量参数附近的光线保持更高积分预算。

## 验证

```bash
python3 scripts/verify_physics.py
python3 scripts/verify_binary_preview.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_nr_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_nr_contract.py
```

Schwarzschild 数值回归覆盖：

- 临界冲量参数 `b_c = 3√3 M`
- 弱场偏折与 `4M/b` 的一致性
- 有限距离观察者的阴影角直径
- 零测地线积分守恒量
- 184 / 288 步实时预算下的捕获与逃逸行为

双黑洞预览检查覆盖 schema 与明确的非 NR 安全标记、等质量对称性、声明的最低阶 PN 方程、轨道间距/相位/过渡权重的单调性、有限参数，以及代表性余留体自旋的 Kerr 上界。一个与 shader 方程一致的 90×45 CPU 光线网格还会验证：代表性的宽轨道、合并过渡与余留体视角在固定 512 步生产预算下均不存在未收敛光线。这些测试只验证内部一致性和明确的实时收敛门槛，不验证双黑洞时空、强场透镜或合并物理。

NR 契约检查严格 JSON/schema、一致的 sidecar 哈希与大小、可移植 artifact
定位规则、连续有序 chunk、物理系统与 source/protocol 时间声明、互逆的空间
仿射标架、正确的 ICRS 旋转、观测者 tetrad 正交归一性、光线积分与边界语义，
以及合法且有限的逐光线结果。数据声明可渲染前，还会把 outcome fractions
与解码记录交叉核对。未知或缺失字段、重复键、非有限数、路径越界和含混的
无效光线状态会被拒绝。验证通过表示 **protocol-conformant**，不表示
**NR-backed** 或 **physically validated**。

两套脚本共同覆盖一组明确的数值性质与架构契约，但不等价于完整画面、辐射模型或所有 GPU 的自动化验证。当前仓库尚未配置 GPU 图像回归 CI。

## 天空素材与署名

- **ESA/Gaia/DPAC · A. Moitinho**：可选 16000×8000 Gaia EDR3 全天图，CC BY-SA 3.0 IGO。
- **ESO/S. Brunier**：仓库内置 6000×3000 银河摄影背景，CC BY 4.0。
- `assets/deep-field.webp`：由仓库脚本生成的备用深空素材，不是默认背景。

下载地址、处理方式、哈希和完整许可信息见 [`assets/SOURCES.md`](./assets/SOURCES.md)。第三方素材不会因本项目代码未来采用某种许可证而被重新授权。

## 许可证

当前仓库尚未声明项目代码许可证。第三方天空素材与 vendored 依赖仍分别遵循其原始许可；在选择项目代码许可证前，请不要假设仓库内容已按 MIT、Apache-2.0 等许可证授权。
