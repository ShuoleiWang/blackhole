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

右上角会显示实际后端、GPU、HDR/色域模式、FPS 与内部渲染分辨率。显示器进入 HDR 时会显示 `HDR · P3 · FP16`；若窗口所在屏幕只提供 SDR，则同一扩展管线会如实显示 `P3 扩展 · 屏幕 SDR`。画质会在用户设置的上限内动态调整，以维持交互帧率。

## macOS HDR 输出

WebGPU 路径优先请求 `rgba16float`、`display-p3` 与 `toneMapping: extended`。后处理在 Display‑P3 中保留最高约 4 倍 SDR 漫反射白的高光，不做 1.0 上限裁切，也不做 8 位抖动；macOS 合成器负责将扩展值映射到当前显示器的可用 EDR 余量。Safari 还会使用 `dynamic-range-limit: no-limit` 渐进增强。

扩展 HDR 不等于强行提高全画面亮度：银河漫反射背景保持在纸白以下，吸积盘内缘和亮星才进入 HDR 高光。WebGL2 是明确标注的 sRGB/SDR 回退，不伪装成 HDR。

## 交互

- 鼠标拖动 / 单指拖动：改变轨道平面和观测相位
- 滚轮 / 双指缩放：改变观测半径
- 方向键：微调视角；`+` / `-`：改变观测半径；空格：暂停或继续
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
- 盘面不是不透明色带：有限寿命的二维 MRI 湍流单元随局部 Kepler 角速度平流，发射采用视线光学深度、覆盖率和肢暗化，并累积一条光线的多次盘面交叉
- 背景在世界天球上保持固定，仅由逃逸光线方向采样；引力透镜因此自然复制、拉伸并环化银河结构
- 解析恒星同样只在逃逸方向上生成，其黑体色温还会随观测者频移改变；不是后期贴在屏幕上的点。临界曲线使用局部 2×2 测地线覆盖采样，并只在窄临界带剥离底图中已烘焙的摄影 PSF 星核

### 为什么不是整圈纯白或整圈同一种金色

吸积盘没有唯一固定颜色；它由质量、吸积率、半径、观测波段和相对论频移共同决定。默认超大质量、低吸积率薄盘的峰值约 4500 K，因此本征颜色是太阳金/暖白。接近观测者的一侧会被 Doppler 增亮并蓝移到黄白色，远离侧则更暗、更橙红，这个不对称才是保留完整频移后的物理结果。

电影《星际穿越》的盘由美术人员设为处处 4500 K，并为了大众可读性关闭了强烈的 Doppler 颜色与亮度不对称，再加入 IMAX 镜头眩光。本项目采用相近的默认温标，但不关闭这些物理效应，也没有给盘乘统一金色滤镜。

这是数值实时模型，而不是无限精度求解器。当前有意采用物理自洽的非旋转 Schwarzschild 度规；没有把不完整的“自旋特效”冒充 Kerr 光线追踪。最靠近临界光子轨道、需要超过 384 个角步长的指数级窄区域会受实时步数上限影响。

## 360° 银河素材

默认全天球来自 ESA/Gaia EDR3 的 **The colour of the sky** 等距柱状投影，直接使用官方未缩放的 16000×8000 PNG（约 1.28 亿像素），由超过 18 亿颗恒星的数据生成。WebGPU 会显式请求 Metal 暴露的 16K 原生纹理上限；不支持 16K 的 GPU 自动回退到 ESO/S. Brunier 的官方 6000×3000 全天摄影，再回退到 4096×2048 WebP。完整来源、许可、处理说明与哈希见 [`assets/SOURCES.md`](./assets/SOURCES.md)。Gaia 官方页面：<https://sci.esa.int/web/gaia/-/the-colour-of-the-sky-from-gaia-s-early-data-release-3-equirectangular-projection>。

`assets/deep-field.webp` 与 `scripts/generate_deep_field.py` 是可复现的自生成深空备用素材，不是默认背景。

## 验证

```bash
python3 scripts/verify_physics.py
```

回归检查覆盖临界阴影、有限距离阴影角、弱场偏折和积分守恒量。桌面与移动视口的最新截图位于 `artifacts/`。

主要代码：

- `src/shaders.js`：WGSL / GLSL 的测地线、薄盘辐射与 HDR 后处理
- `src/webgpu-renderer.js`：WebGPU / Metal 双通道渲染管线
- `src/webgl-renderer.js`：WebGL2 GPU 回退管线
- `src/main.js`：物理参数、相机轨道、鼠标/触控和动态画质
