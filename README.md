# Schwarzschild 深空观测台

一个直接在浏览器 GPU 上运行的实时黑洞观测模拟。程序优先使用 **WebGPU**；在 macOS 上由浏览器的 **Metal** 后端驱动 Apple GPU，并优先建立 **16 位浮点 Display‑P3 扩展 HDR** 交换链。若 WebGPU 不可用，会自动切换到 WebGL2 硬件加速路径。

画面不是屏幕空间扭曲：每个像素都从运动观测者的局部标架反向追踪一条 Schwarzschild 零测地线，同一条光线同时决定事件视界捕获、吸积盘命中和 360° 银河背景的逃逸方向。

## 运行

项目无需构建：

```bash
./scripts/fetch_gaia_sky.sh
python3 -m http.server 4173
```

首次运行先执行下载脚本。它从 ESA 官方源恢复未降采样的 16000×8000
Gaia 全天图，并在安装前核对固定 SHA‑256；原图不存入 Git 仓库。随后打开
`http://localhost:4173`。WebGPU 需要 localhost 或 HTTPS 安全上下文。

- 默认：优先 WebGPU / Metal
- 调试回退：`http://localhost:4173/?renderer=webgl`
- 强制关闭扩展 HDR：`http://localhost:4173/?hdr=0`
- 保持 6K 快速背景：`http://localhost:4173/?sky=high`
- 阻塞等待 16K 背景（诊断用）：`http://localhost:4173/?sky=ultra`

右上角会显示实际后端、GPU、HDR/色域模式、FPS 与内部渲染分辨率。显示器进入 HDR 时会显示 `HDR · P3 · FP16`；若窗口所在屏幕只提供 SDR，则同一扩展管线会如实显示 `P3 扩展 · 屏幕 SDR`。画质会在用户设置的上限内动态调整，以维持交互帧率。

## macOS HDR 输出

WebGPU 路径优先请求 `rgba16float`、`display-p3` 与 `toneMapping: extended`。后处理在 Display‑P3 中保留最高约 4 倍 SDR 漫反射白的高光，不做 1.0 上限裁切，也不做 8 位抖动；macOS 合成器负责将扩展值映射到当前显示器的可用 EDR 余量。Safari 还会使用 `dynamic-range-limit: no-limit` 渐进增强。

扩展 HDR 不等于强行提高全画面亮度：银河漫反射背景保持在纸白以下，吸积盘内缘和亮星才进入 HDR 高光。WebGL2 是明确标注的 sRGB/SDR 回退，不伪装成 HDR。

## M3 Pro / M4 兼容性

渲染器不按芯片名称分支，也不依赖 M4 独有指令。M3 与 M4 均属于 Apple9 GPU family，并支持本项目使用的 Metal 3/4、16K 2D 纹理与扩展范围像素格式；程序仍以浏览器实际返回的 WebGPU/WebGL limits、HDR canvas 配置和 RGBA16F framebuffer 完整性作为最终依据。Apple 官方能力表：<https://developer.apple.com/metal/capabilities/>。

- M3 Pro 已实测 WebGPU/Metal、WebGL2/Metal、16K 后台升级、Display‑P3 FP16 HDR 与 SDR 降级。
- M4 走同一 Apple9 能力路径；本轮没有 M4 实机，因此仍建议发布前在目标 M4 浏览器做一次 smoke test。
- 动态画质按真实帧时自适应，不假设 M4 一定比 M3 Pro 快；普通光线会在 184–288 步间调节，临界光子环保持 384 步。
- 16K 解码或 GPU 上传失败时保留已显示的 6K 背景；HDR 配置未被浏览器实际保留时依次降为 Display‑P3 SDR、sRGB SDR，最后才切 WebGL2。

## 交互

- 鼠标拖动 / 单指拖动：改变观测相位和圆轨道所在平面
- 滚轮 / 双指缩放：改变观测半径
- 方向键：微调视角；`0`：令圆轨道与吸积盘共面并严格侧视；`+` / `-`：改变观测半径；空格：暂停或继续
- “科学真色 / 哈勃调色”只改变显示谱段、PSF 与色调，不改变测地线、遮挡或频移
- 质量改变物理半径、盘温度与时间尺度；吸积率改变薄盘温度和辐射通量

## 物理模型

使用几何单位 `G = c = M = 1`：

- 事件视界 `r = 2M`
- 光子球 `r = 3M`
- 临界冲量参数 `b_c = 3√3 M`
- 非旋转薄盘 ISCO `r = 6M`
- 零测地线方程 `u'' = -u + 3u²`，以 Störmer–Verlet 在 GPU fragment shader 中积分
- 相机屏幕光线先从圆轨道共动标架做 Lorentz 变换，再进入局部 Schwarzschild 静态标架
- 吸积盘采用差分 Kepler 转动、零力矩薄盘温度轮廓、Planck 可见波段颜色及 `g⁴` 辐射转移；UI 中的 Eddington 比定义为 `L/L_Edd`，默认峰值温度约 4500 K
- 盘面不是纯色带：有限寿命、无固定旋臂的二维 MRI 湍流单元随局部 Kepler 角速度平流；温度起伏、面密度、视线光学深度、覆盖率和肢暗化分别参与辐射，并累积一条光线的多次盘面交叉
- 背景在世界天球上保持固定，仅由逃逸光线方向采样；引力透镜因此自然复制、拉伸并环化银河结构
- 解析恒星同样只在逃逸方向上生成，其黑体色温还会随观测者频移改变；不是后期贴在屏幕上的点。临界曲线使用局部 2×2 测地线覆盖采样，并只在窄临界带剥离底图中已烘焙的摄影 PSF 星核

### 为什么不是整圈纯白或整圈同一种金色

吸积盘没有唯一固定颜色；它由质量、吸积率、半径、观测波段和相对论频移共同决定。默认超大质量、低吸积率薄盘的峰值约 4500 K，因此本征颜色是太阳金/暖白。接近观测者的一侧会被 Doppler 增亮并蓝移到黄白色，远离侧则更暗、更橙红，这个不对称才是保留完整频移后的物理结果。

电影《星际穿越》的盘由美术人员设为处处 4500 K，并为了大众可读性关闭了强烈的 Doppler 颜色与亮度不对称，再加入 IMAX 镜头眩光。本项目采用相近的默认温标，但不关闭这些物理效应，也没有给盘乘统一金色滤镜。

### 为什么严格侧视时直接像是一条线

当前几何采用经典的**理想零厚度薄盘**。当视线与盘面严格平行时，盘的直接投影在数学上必然退化成一条零测度的线；这不是透视错误。黑洞附近的完整图像仍不应只有直线：远侧盘和盘底发出的光会被 Schwarzschild 引力弯折，在阴影上、下方形成次级像和贴近临界曲线的细环。`0` 键可固定复现这个极限；由于程序逐像素追踪中心光线，它不会人为给这条零测度直接像添加一个会发光的“径向侧壁”。

真实吸积流具有有限尺度高度 `H(R)`；有限厚度但仍几何薄的盘在严格侧视时会是一条窄带，RIAF 或厚盘则可能呈现更厚的环或晕。当前没有用屏幕空间把它假加粗；若继续升级为有限厚度模型，需要沿测地线积分三维发射与吸收，而不是简单扩大二维交点。经典薄盘直接像/次级像可参见 [Luminet 1979](https://ui.adsabs.harvard.edu/abs/1979A%26A....75..228L/abstract)，电影级相对论成像与艺术取舍可参见 [James 等 2015](https://doi.org/10.1088/0264-9381/32/6/065001)。

这是数值实时模型，而不是无限精度求解器。当前有意采用物理自洽的非旋转 Schwarzschild 度规；没有把不完整的“自旋特效”冒充 Kerr 光线追踪。最靠近临界光子轨道、需要超过 384 个角步长的指数级窄区域会受实时步数上限影响。

## 360° 银河素材

默认先显示 ESO/S. Brunier 的 6000×3000 全天摄影，再在后台升级到 ESA/Gaia EDR3 的 **The colour of the sky** 等距柱状投影，即官方未缩放的 16000×8000 PNG（约 1.28 亿像素），由超过 18 亿颗恒星的数据生成。这样首次画面不会被 236 MiB 解码阻塞；WebGPU 会显式请求 Metal 暴露的 16K 原生纹理上限，并在上传前后检查实际 GPU 错误。不支持 16K 或内存不足时会保留 6K，再回退到 4096×2048 WebP。完整来源、许可、处理说明与哈希见 [`assets/SOURCES.md`](./assets/SOURCES.md)。Gaia 官方页面：<https://sci.esa.int/web/gaia/-/the-colour-of-the-sky-from-gaia-s-early-data-release-3-equirectangular-projection>。

`assets/deep-field.webp` 与 `scripts/generate_deep_field.py` 是可复现的自生成深空备用素材，不是默认背景。

## 验证

```bash
python3 scripts/verify_physics.py
```

回归检查覆盖临界阴影、有限距离阴影角、弱场偏折和积分守恒量。

主要代码：

- `src/shaders.js`：WGSL / GLSL 的测地线、薄盘辐射与 HDR 后处理
- `src/webgpu-renderer.js`：WebGPU / Metal 双通道渲染管线
- `src/webgl-renderer.js`：WebGL2 GPU 回退管线
- `src/main.js`：物理参数、相机轨道、鼠标/触控和动态画质
