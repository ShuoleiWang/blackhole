export const DEFAULT_LOCALE = "en";
export const SUPPORTED_LOCALES = Object.freeze(["en", "zh-CN"]);

const EN = Object.freeze({
  "meta.description": "Interactive real-time relativistic ray tracing for black-hole systems on Apple M3 Pro.",
  "document.defaultTitle": "Real-time Relativistic Black-hole Ray Tracing",
  "app.ariaLabel": "Relativistic black-hole observatory simulation",
  "canvas.ariaLabel": "Draggable black-hole and deep-space observation view",
  "status.ariaLabel": "Live rendering status",
  "status.backend": "Compute backend",
  "status.initializing": "Initializing",
  "status.gpu": "Graphics processor",
  "status.detecting": "Detecting",
  "status.output": "Display output",
  "status.fps": "Frame rate",
  "status.renderScale": "Actual scale",
  "panel.defaultEyebrow": "GPU null-geodesic imaging",
  "panel.defaultTitle": "Schwarzschild black hole",
  "panel.actions": "View controls",
  "motion.pauseOrbit": "Pause physical orbit",
  "motion.resumeOrbit": "Resume physical orbit",
  "view.reset": "Reset observation view",
  "panel.parameters": "Parameters",
  "panel.observationSettings": "observation settings",
  "panel.expandObservationSettings": "Expand observation settings",
  "panel.expand": "Expand {context}",
  "panel.collapse": "Collapse {context}",
  "scene.navigation": "Spacetime scene",
  "scene.binary": "Vacuum binary",
  "scene.binaryDualDisk": "Dual-disk binary",
  "scene.single": "Single black hole",
  "scene.reference": "Science reference",
  "reference.navigation": "Fixed-camera offline calibration reference",
  "reference.schwarzschildLink": "Schwarzschild (non-rotating)",
  "reference.kerrLink": "Kerr (rotating remnant)",
  "mode.group": "Imaging color mode",
  "mode.science": "Scientific true color",
  "mode.hubble": "Hubble palette",
  "mode.frequency": "Frequency shift g",
  "diagnostics.advanced": "Advanced numerical diagnostics",
  "diagnostics.lookback": "Coordinate lookback time",
  "diagnostics.lookbackTitle": "A coordinate quantity, not an observable relative arrival-time delay",
  "diagnostics.null": "Null residual",
  "diagnostics.nullTitle": "Maximum null-condition residual recorded along the ray",
  "diagnostics.error": "Projection error",
  "diagnostics.errorTitle": "Numerical projection error in detector pixels",
  "readout.group": "Physical observation readouts",
  "readout.observer": "Observer position",
  "readout.schwarzschildRadius": "Schwarzschild radius",
  "readout.shadowDiameter": "Shadow angular diameter",
  "inspector.title": "Pixel ray record",
  "inspector.coordinatesEmpty": "Pixel x — · y —",
  "inspector.close": "Close pixel ray record",
  "inspector.direction": "Escape direction (ICRS)",
  "inspector.frequency": "Frequency shift g",
  "inspector.lookback": "Coordinate lookback time",
  "inspector.outcome": "Ray outcome",
  "inspector.details": "Numerical and ABI details",
  "inspector.null": "Null residual",
  "inspector.error": "Projection error",
  "inspector.validity": "Validity mask",
  "inspector.raw": "Raw little-endian bytes",
  "inspector.help": "Click the view to select a pixel. With the canvas focused, use arrow keys to move one pixel, hold Shift to move faster, and press Esc or the close button to exit.",
  "binary.timeline": "Binary black-hole timeline",
  "binary.waveformPreview": "Strain preview · r h₊ / M",
  "binary.waveformAria": "Manifest-sampled binary black-hole strain preview",
  "binary.transport": "Binary black-hole time controls",
  "binary.pauseTimeline": "Pause binary black-hole timeline",
  "binary.resumeTimeline": "Resume binary black-hole timeline",
  "binary.simulationTime": "Simulation time",
  "binary.slowMotionTitle": "Changes playback speed only; physical data remain unchanged",
  "binary.slowMotion": "Merger slow motion",
  "binary.actualRateZero": "Actual 0 M/s",
  "binary.ariaTime": "{time}, {regime}",
  "parameters.title": "Observation parameters",
  "parameters.realtime": "Real time",
  "control.mass": "Black-hole mass",
  "control.accretion": "Accretion rate",
  "control.exposure": "Exposure",
  "control.timeScale": "Time scale",
  "control.quality": "Quality ceiling",
  "sky.label": "Sky source",
  "sky.eso": "ESO native 6000×3000 (locked)",
  "sky.gaia": "Gaia native 16000×8000 (locked)",
  "sky.hint": "Always uploaded at native dimensions, with no downsampling or silent fallback. The Gaia file is about 236 MB and decodes to about 488 MiB of unified-memory GPU texture data.",
  "physics.schwarzschildHtml": "Uses the non-rotating Schwarzschild metric. Imaging modes change only spectral mapping and telescope response; geodesics, occlusion, and frequency shift remain unchanged. The default disk peaks near 4500 K, with color computed from blackbody emission and relativistic frequency shift rather than a gold filter. All-sky data: <a href=\"https://sci.esa.int/web/gaia/-/the-colour-of-the-sky-from-gaia-s-early-data-release-3-equirectangular-projection\" target=\"_blank\" rel=\"noreferrer\">ESA/Gaia/DPAC · A. Moitinho</a>; 6K photographic fallback: <a href=\"https://www.eso.org/public/images/eso0932a/\" target=\"_blank\" rel=\"noreferrer\">ESO/S. Brunier</a>.",
  "hint.desktop": "Drag to orbit · Wheel to zoom · Press 0 for side view",
  "hint.touch": "One finger to orbit · Two fingers to zoom",
  "language.label": "Interface language",
  "fatal.initialization": "Initialization failed",
  "fatal.rendererTitle": "Unable to start GPU renderer",
  "fatal.dataValidation": "Data validation failed",
  "fallback.reason": "WebGPU fallback reason: {reason}",
  "binary.regime.inspiral": "SXS NR inspiral",
  "binary.regime.gap": "A/B tracks ended · event gap",
  "binary.regime.merger": "Common horizon formed",
  "binary.regime.ringdown": "Remnant ringdown",
  "binary.mode.sky": "Sky image",
  "binary.mode.weakField": "Weak-field preview",
  "binary.mode.outcome": "Ray outcome",
  "binary.mode.outcomeTitle": "Blue is captured, green is escaped, and magenta is unresolved; classifications come from the current WebGPU ray integration",
  "binary.mode.frequency": "Frequency shift g",
  "binary.mode.frequencyTitle": "Ratio g of observed frequency to frequency at infinity; physically meaningful only for escaped rays",
  "binary.mode.lookback": "Coordinate lookback time",
  "binary.mode.lookbackTitle": "Coordinate time integrated along a fast-light slice, not an observable relative arrival-time delay",
  "binary.mode.null": "Null / H residual",
  "binary.mode.nullTitle": "Maximum normalized null-Hamiltonian residual recorded along the ray",
  "binary.mode.cost": "Integration-step cost",
  "binary.mode.costTitle": "Executed integration steps as a fraction of the compiled 320-step limit; this is computational cost, not a physical quantity",
  "binary.status.strongTrace": "Real-time 3+1 Hamiltonian strong-field ray tracing",
  "binary.status.fastLight": "fast-light approximation · not full NR",
  "binary.status.advanced": "Advanced diagnostics: lookback time / null residual / integration cost",
  "binary.status.compatibility": "Compatibility fallback",
  "binary.status.legacy": "Legacy two-centre weak-field preview",
  "binary.status.noParity": "No strong-field physical parity with WebGPU",
  "binary.status.hiddenDiagnostics": "Advanced strong-field numerical diagnostics hidden",
  "binary.playback.endHold": "End hold · 0 M/s",
  "binary.playback.paused": "Paused · 0 M/s",
  "binary.playback.actual": "Actual {rate} M/s",
  "binary.playback.actualWithFactor": "Actual {rate} M/s · {factor}×",
  "binary.playback.slowOn": "Merger slow motion on",
  "binary.playback.slowOff": "Merger slow motion off",
  "binary.readout.gap": "A/B tracks ended · common-horizon event not yet reached",
  "binary.readout.horizon": "Common horizon formed · A/B tracks ended",
  "binary.readout.remnant": "Single SXS remnant",
  "binary.documentTitle": "Live binary black holes · Deep-space observatory",
  "binary.eyebrow": "Real-time strong-field ray tracing · SXS anchored",
  "binary.title": "Live binary black holes",
  "binary.observerLabel": "SXS coordinate evidence (does not drive ray tracing)",
  "binary.radiusLabel": "1 M (GM/c²)",
  "binary.segmentLabel": "Data segment",
  "binary.massLabel": "Total system mass",
  "binary.initialStatus.strong": "WebGPU strong-field production path",
  "binary.initialStatus.anchor": "SXS h₂₂ / merger-event anchored",
  "binary.initialStatus.fallback": "WebGL2 falls back to the legacy weak field",
  "binary.desktopHint": "Drag to observe · Wheel to zoom · Scrub the timeline · Space to pause",
  "binary.physicsHtml": "The waveform, common-horizon time, and remnant parameters are anchored to {sourceLink}. The horizon-centroid separation and phase shown at right are <strong>gauge-dependent coordinate evidence and never drive WebGPU black-hole positions</strong>; the real-time orbit is generated from the h₂₂ frequency and a PN/EOB-like quasi-circular relation, while rays are integrated in a boosted superposed Kerr–Schild 3+1 metric. This is <strong>a strong-field fast-light approximation, not a constraint-solved full NR spacetime or a slow-light solution</strong>; WebGL2 explicitly falls back to the legacy weak-field preview. Merger slow motion changes wall-clock playback speed only.",
  "binary.vacuum": "Vacuum",
  "dualDisk.documentTitle": "Dual-disk binary · Deep-space observatory",
  "dualDisk.eyebrow": "Exploratory emission · SXS-anchored lensing",
  "dualDisk.title": "Dual-disk binary black holes",
  "dualDisk.observerLabel": "SXS coordinate evidence (does not drive ray tracing)",
  "dualDisk.radiusLabel": "Mini-disk outer radii",
  "dualDisk.segmentLabel": "Emission state",
  "dualDisk.massLabel": "Total system mass",
  "dualDisk.accretionLabel": "Per-disk emission proxy",
  "dualDisk.accretionValue": "{rate}% Edd proxy",
  "dualDisk.accretionAriaValue": "{rate}% Eddington emission proxy per disk",
  "dualDisk.mode.sky": "Mini-disks + sky",
  "dualDisk.mode.skyTitle": "CIE visible-band thin mini-disk emission composited with the lensed native-resolution sky",
  "dualDisk.mode.weakField": "Weak-field preview · no disk parity",
  "dualDisk.readout.radiiLabel": "Mini-disk outer radii",
  "dualDisk.readout.radiiValue": "A {radiusA} M · B {radiusB} M",
  "dualDisk.readout.emissionLabel": "Emission state",
  "dualDisk.readout.emissionActive": "Two idealized mini-disks active",
  "dualDisk.readout.emissionFading": "Fading as the Roche/ISCO stable annulus contracts",
  "dualDisk.readout.emissionTidallyDisrupted": "No stable mini-disk annulus · emission off",
  "dualDisk.readout.emissionUnmodeled": "Post-merger emission unmodeled",
  "dualDisk.readout.emissionUnavailable": "Dual-disk emission unavailable in WebGL2 preview",
  "dualDisk.initialStatus.strong": "WebGPU 3+1 strong-field fast-light",
  "dualDisk.initialStatus.emission": "Idealized thin mini-disks",
  "dualDisk.initialStatus.boundary": "No GRMHD or self-consistent radiative transfer · not full NR",
  "dualDisk.initialStatus.fallback": "WebGL2 is a weak-field preview with no dual-disk physical parity",
  "dualDisk.status.tidalShutdown": "C² tidal shutdown when no Roche/ISCO stable annulus remains",
  "dualDisk.status.postMerger": "Post-merger emission unmodeled",
  "dualDisk.physicsHtml": "The waveform, common-horizon time, remnant parameters, and strong-field lensing are anchored to the same declared {sourceLink} binary contract as the vacuum scene. The two luminous surfaces are <strong>idealized geometrically thin mini-disks controlled by a visualization proxy</strong>. Their true-colour proxy integrates the 380–780 nm blackbody response through a fixed CIE observer, while the C² photospheric edge and low-amplitude, unit-mean tidal/emissivity structure remain analytic prescriptions, not SXS matter data. The model does not include GRMHD, volumetric absorption, polarization, or self-consistent spectral radiative transfer. As Roche-lobe truncation squeezes each outer edge toward the ISCO, emission closes with a C² transition when no stable annulus remains; it is strictly zero after the common horizon, and post-merger emission is unmodeled. This remains a <strong>strong-field fast-light approximation, not full NR</strong>. WebGL2 is only a weak-field compatibility preview and makes no dual-disk physical-parity claim.",
  "reference.schwarzschildTitle": "Schwarzschild offline calibration",
  "reference.kerrTitle": "Kerr-remnant offline calibration",
  "reference.mode.composite": "Composite image",
  "reference.mode.outcome": "Ray outcome",
  "reference.mode.lookback": "Coordinate lookback time",
  "reference.mode.frequency": "Frequency shift g",
  "reference.mode.null": "Null residual",
  "reference.mode.error": "Projection error",
  "reference.progress.manifestVerified": "{title} manifest SHA-256 verified",
  "reference.progress.manifestFetching": "Fetching pinned manifest for {title}…",
  "reference.progress.sidecarVerified": "Manifest sidecar cross-verified",
  "reference.progress.sidecarChecking": "Cross-checking manifest sidecar…",
  "reference.progress.chunks": "Verifying transfer-map chunks {completed}/{total}…",
  "reference.progress.decoded": "Strictly decoded {total} ray records",
  "reference.progress.default": "Verifying stationary transfer map…",
  "reference.outcomes": "{captured}% captured · {escaped}% escaped",
  "reference.loadFailed": "Transfer-map validation failed: {message}",
  "reference.retry": "Retry validation",
  "reference.returnBinary": "Return to live binary",
  "reference.panelLabel": "display settings",
  "reference.documentTitle": "{title} · Deep-space observatory",
  "reference.canvasAria": "{title} fixed-camera offline calibration view; click a pixel or use arrow keys to inspect its ray record",
  "reference.eyebrow": "Research tool · fixed-camera offline calibration",
  "reference.fixedObserver": "Fixed observer camera",
  "reference.spin": "Dimensionless spin",
  "reference.rayOutcome": "Ray outcome",
  "reference.massScale": "Mass scale",
  "reference.displaySettings": "Display settings",
  "reference.fixedData": "Fixed data",
  "reference.status.analytic": "Analytic vacuum reference",
  "reference.status.fixedCamera": "Fixed camera",
  "reference.status.chunks": "{verified}/{total} data-chunk SHA-256 hashes verified",
  "reference.modeGroup": "Fixed-camera offline calibration diagnostic view",
  "reference.desktopHint": "Click to inspect a ray record · Arrow keys to move · Shift to accelerate",
  "reference.touchHint": "Tap the view to inspect a pixel ray record",
  "reference.description.kerr": "Analytic Kerr-remnant vacuum spacetime",
  "reference.description.schwarzschild": "Analytic Schwarzschild vacuum spacetime",
  "reference.physics": "{description} in a fixed-camera offline calibration view, used only to research and validate the transfer-map data chain; this is not a binary black-hole merger image, NR ray tracing, or a high-fidelity final render, and it includes no accretion emission. Coordinates: {coordinates}. Integrator: {integrator}.",
  "reference.vacuum": "Vacuum · no emission model",
  "reference.static": "Static",
  "reference.fixedTetrad": "fixed orthonormal tetrad",
  "reference.pixel": "Pixel x {x} · y {y}",
  "reference.loadingEyebrow": "Pinned trust root · validating",
  "reference.outcome.captured": "captured",
  "reference.outcome.escaped": "escaped",
  "reference.outcome.unresolved": "unresolved",
  "reference.outcome.outsideDomain": "outside domain",
  "reference.outcome.integratorFailure": "integrator failure",
  "reference.outcome.missing": "missing",
  "reference.outcome.unusable": "unusable",
  "reference.none": "none",
  "renderer.fallback.deviceLost": "WebGPU device connection lost",
  "renderer.fallback.renderError": "WebGPU rendering error",
  "renderer.webglMetalFallback": "WebGL2 · Metal fallback",
  "renderer.appleGpu": "Apple GPU",
  "renderer.hardwareGpu": "Hardware GPU",
  "renderer.appleGpuMetal": "Apple GPU · Metal",
  "renderer.highPerformanceGpu": "High-performance GPU",
  "renderer.srgb": "sRGB standard dynamic range",
  "renderer.skyPending": "Galactic background pending",
  "renderer.p3ExtendedSdr": "P3 extended · display SDR",
  "renderer.hdrDescription": "16-bit float Display-P3 extended HDR (highlights up to 4× SDR white)",
  "renderer.p3Sdr": "Display-P3 standard dynamic range",
  "renderer.skyDetail": "{width}×{height} native panorama · resolved stellar layer",
  "renderer.webglSdr": "WebGL2 fallback uses sRGB standard dynamic range",
  "renderer.webglSdrIntermediate": "WebGL2 fallback uses sRGB standard dynamic range · {intermediate} intermediate buffer",
});

const ZH_CN = Object.freeze({
  "meta.description": "面向 Apple M3 Pro 的可交互相对论黑洞系统实时光线追踪。",
  "document.defaultTitle": "相对论黑洞实时光线追踪",
  "app.ariaLabel": "相对论黑洞观测模拟",
  "canvas.ariaLabel": "可拖动的黑洞与深空观测画面",
  "status.ariaLabel": "实时渲染状态",
  "status.backend": "计算后端",
  "status.initializing": "正在初始化",
  "status.gpu": "图形处理器",
  "status.detecting": "检测中",
  "status.output": "显示输出",
  "status.fps": "帧率",
  "status.renderScale": "实际倍率",
  "panel.defaultEyebrow": "GPU 零测地线实时成像",
  "panel.defaultTitle": "Schwarzschild 黑洞",
  "panel.actions": "视角操作",
  "motion.pauseOrbit": "暂停物理轨道",
  "motion.resumeOrbit": "继续物理轨道",
  "view.reset": "重置观测视角",
  "panel.parameters": "参数",
  "panel.observationSettings": "观测参数",
  "panel.expandObservationSettings": "展开观测参数",
  "panel.expand": "展开{context}",
  "panel.collapse": "收起{context}",
  "scene.navigation": "时空场景",
  "scene.binary": "真空双黑洞",
  "scene.binaryDualDisk": "双吸积盘双黑洞",
  "scene.single": "单黑洞",
  "scene.reference": "科学参考",
  "reference.navigation": "固定相机离线校准参考",
  "reference.schwarzschildLink": "Schwarzschild（非旋转）",
  "reference.kerrLink": "Kerr（旋转余留体）",
  "mode.group": "成像色彩模式",
  "mode.science": "科学真色",
  "mode.hubble": "哈勃调色",
  "mode.frequency": "频移因子 g",
  "diagnostics.advanced": "高级数值诊断",
  "diagnostics.lookback": "坐标回溯时间",
  "diagnostics.lookbackTitle": "坐标量，不等同于可观测的相对到达时延",
  "diagnostics.null": "零性残差",
  "diagnostics.nullTitle": "沿光线记录的最大零性条件残差",
  "diagnostics.error": "投影误差",
  "diagnostics.errorTitle": "以探测器像素为单位的数值投影误差",
  "readout.group": "物理观测读数",
  "readout.observer": "观测位置",
  "readout.schwarzschildRadius": "史瓦西半径",
  "readout.shadowDiameter": "阴影角直径",
  "inspector.title": "像素光线记录",
  "inspector.coordinatesEmpty": "像素 x — · y —",
  "inspector.close": "关闭像素光线记录",
  "inspector.direction": "逃逸方向（ICRS）",
  "inspector.frequency": "频移因子 g",
  "inspector.lookback": "坐标回溯时间",
  "inspector.outcome": "光线结果",
  "inspector.details": "数值与 ABI 详情",
  "inspector.null": "零性残差",
  "inspector.error": "投影误差",
  "inspector.validity": "有效位掩码",
  "inspector.raw": "原始小端序字节",
  "inspector.help": "点击画面选择像素；聚焦画布后用方向键逐像素移动，Shift 加速，关闭按钮或 Esc 退出。",
  "binary.timeline": "双黑洞时间线",
  "binary.waveformPreview": "应变预览 · r h₊ / M",
  "binary.waveformAria": "基于清单采样的双黑洞应变预览",
  "binary.transport": "双黑洞时间控制",
  "binary.pauseTimeline": "暂停双黑洞时间线",
  "binary.resumeTimeline": "继续双黑洞时间线",
  "binary.simulationTime": "模拟时间",
  "binary.slowMotionTitle": "只改变播放速度，不改变物理数据",
  "binary.slowMotion": "合并慢放",
  "binary.actualRateZero": "实际 0 M/s",
  "binary.ariaTime": "{time}，{regime}",
  "parameters.title": "观测参数",
  "parameters.realtime": "实时",
  "control.mass": "黑洞质量",
  "control.accretion": "吸积率",
  "control.exposure": "曝光",
  "control.timeScale": "时间倍率",
  "control.quality": "画质上限",
  "sky.label": "天空素材",
  "sky.eso": "ESO 原始 6000×3000（锁定）",
  "sky.gaia": "Gaia 原始 16000×8000（锁定）",
  "sky.hint": "始终按原始尺寸上传，不降采样或静默回退。Gaia 文件约 236 MB，解码为 GPU 纹理约占 488 MiB 统一内存。",
  "physics.schwarzschildHtml": "采用非旋转 Schwarzschild 度规。成像模式只改变谱段映射与望远镜响应；测地线、遮挡和频移保持一致。默认盘峰值约 4500 K，颜色由黑体谱与相对论频移计算，并非金色滤镜。全天球数据：<a href=\"https://sci.esa.int/web/gaia/-/the-colour-of-the-sky-from-gaia-s-early-data-release-3-equirectangular-projection\" target=\"_blank\" rel=\"noreferrer\">ESA/Gaia/DPAC · A. Moitinho</a>；6K 摄影回退：<a href=\"https://www.eso.org/public/images/eso0932a/\" target=\"_blank\" rel=\"noreferrer\">ESO/S. Brunier</a>。",
  "hint.desktop": "拖动环绕 · 滚轮缩放 · 0 键侧视",
  "hint.touch": "单指环绕 · 双指缩放",
  "language.label": "界面语言",
  "fatal.initialization": "初始化失败",
  "fatal.rendererTitle": "无法启动 GPU 渲染器",
  "fatal.dataValidation": "数据验证失败",
  "fallback.reason": "WebGPU 回退原因：{reason}",
  "binary.regime.inspiral": "SXS NR 螺旋靠近",
  "binary.regime.gap": "A/B 轨迹结束 · 事件间隙",
  "binary.regime.merger": "共同视界形成",
  "binary.regime.ringdown": "余留体 ringdown",
  "binary.mode.sky": "天空成像",
  "binary.mode.weakField": "弱场预览",
  "binary.mode.outcome": "光线结果",
  "binary.mode.outcomeTitle": "蓝色为捕获，绿色为逃逸，洋红为未收敛；分类来自当前 WebGPU 光线积分",
  "binary.mode.frequency": "频移因子 g",
  "binary.mode.frequencyTitle": "观测频率与无穷远频率之比 g；仅对已逃逸光线有物理意义",
  "binary.mode.lookback": "坐标回溯时间",
  "binary.mode.lookbackTitle": "沿 fast-light 切片积分的坐标时间，不是可观测的相对到达时延",
  "binary.mode.null": "零性 / H 残差",
  "binary.mode.nullTitle": "沿光线记录的最大归一化零 Hamiltonian 残差",
  "binary.mode.cost": "积分步数成本",
  "binary.mode.costTitle": "已执行积分步数相对 320 步编译上限的比例；这是计算成本，不是物理量",
  "binary.status.strongTrace": "实时 3+1 Hamiltonian 强场光追",
  "binary.status.fastLight": "fast-light 近似 · 非完整 NR",
  "binary.status.advanced": "高级诊断：回溯时间 / 零性残差 / 积分成本",
  "binary.status.compatibility": "兼容性回退",
  "binary.status.legacy": "旧 two-centre weak-field 预览",
  "binary.status.noParity": "不具备 WebGPU 强场物理等价性",
  "binary.status.hiddenDiagnostics": "强场高级数值诊断已隐藏",
  "binary.playback.endHold": "末尾停留 · 0 M/s",
  "binary.playback.paused": "已暂停 · 0 M/s",
  "binary.playback.actual": "实际 {rate} M/s",
  "binary.playback.actualWithFactor": "实际 {rate} M/s · {factor}×",
  "binary.playback.slowOn": "合并慢放 开",
  "binary.playback.slowOff": "合并慢放 关",
  "binary.readout.gap": "A/B 轨迹已结束 · 共同视界事件尚未发生",
  "binary.readout.horizon": "共同视界形成 · A/B 轨迹已结束",
  "binary.readout.remnant": "单一 SXS 余留体",
  "binary.documentTitle": "实时双黑洞 · 深空观测台",
  "binary.eyebrow": "实时强场光追 · SXS 锚定",
  "binary.title": "实时双黑洞",
  "binary.observerLabel": "SXS 坐标证据（不驱动光追）",
  "binary.radiusLabel": "1 M（GM/c²）",
  "binary.segmentLabel": "数据区段",
  "binary.massLabel": "系统总质量",
  "binary.initialStatus.strong": "WebGPU 强场生产路径",
  "binary.initialStatus.anchor": "SXS h₂₂ / 合并事件锚定",
  "binary.initialStatus.fallback": "WebGL2 回退为旧弱场",
  "binary.desktopHint": "拖动观察 · 滚轮缩放 · 拖动时间轴 · 空格暂停",
  "binary.physicsHtml": "波形、共同视界时刻和余留体参数锚定到 {sourceLink}。右侧显示的视界质心分离/相位是<strong>依赖规范的坐标证据，绝不进入 WebGPU 黑洞位置</strong>；实时轨道由 h₂₂ 频率与 PN/EOB-like 准圆关系生成，光线在 boosted superposed Kerr–Schild 3+1 度规中积分。这是<strong>强场 fast-light 近似，不是约束求解后的完整 NR 时空，也不是 slow-light</strong>；WebGL2 会明确退回旧 weak-field 预览。合并慢放只改变播放墙钟速度。",
  "binary.vacuum": "真空",
  "dualDisk.documentTitle": "双吸积盘双黑洞 · 深空观测台",
  "dualDisk.eyebrow": "探索性发射 · SXS 锚定透镜",
  "dualDisk.title": "双吸积盘双黑洞",
  "dualDisk.observerLabel": "SXS 坐标证据（不驱动光追）",
  "dualDisk.radiusLabel": "微型盘外半径",
  "dualDisk.segmentLabel": "发射状态",
  "dualDisk.massLabel": "系统总质量",
  "dualDisk.accretionLabel": "单盘发射强度参数",
  "dualDisk.accretionValue": "{rate}% Edd 参数",
  "dualDisk.accretionAriaValue": "每个盘 {rate}% Eddington 发射强度参数",
  "dualDisk.mode.sky": "微型盘 + 天空",
  "dualDisk.mode.skyTitle": "CIE 可见波段薄微型盘发射与原生分辨率透镜天空的合成画面",
  "dualDisk.mode.weakField": "弱场预览 · 不具备吸积盘等价性",
  "dualDisk.readout.radiiLabel": "微型盘外半径",
  "dualDisk.readout.radiiValue": "A {radiusA} M · B {radiusB} M",
  "dualDisk.readout.emissionLabel": "发射状态",
  "dualDisk.readout.emissionActive": "两个理想化微型盘启用",
  "dualDisk.readout.emissionFading": "随 Roche/ISCO 稳定环带收缩而淡出",
  "dualDisk.readout.emissionTidallyDisrupted": "无稳定微型盘环带 · 发射关闭",
  "dualDisk.readout.emissionUnmodeled": "合并后发射未建模",
  "dualDisk.readout.emissionUnavailable": "WebGL2 预览不提供双盘发射",
  "dualDisk.initialStatus.strong": "WebGPU 3+1 强场 fast-light",
  "dualDisk.initialStatus.emission": "理想化薄微型盘",
  "dualDisk.initialStatus.boundary": "无 GRMHD 或自洽辐射转移 · 非完整 NR",
  "dualDisk.initialStatus.fallback": "WebGL2 是弱场预览，不具备双盘物理等价性",
  "dualDisk.status.tidalShutdown": "Roche/ISCO 稳定环带消失时进行 C² 潮汐关闭",
  "dualDisk.status.postMerger": "合并后发射未建模",
  "dualDisk.physicsHtml": "波形、共同视界时刻、余留体参数与强场透镜沿用真空场景声明的 {sourceLink} 双黑洞契约。两个发光表面是<strong>由可视化参数控制的理想化几何薄微型盘</strong>。真彩色代理通过固定 CIE 观察者积分 380–780 nm 黑体响应；C² 光球边缘以及低幅、单位均值的潮汐/发射率结构仍是解析处方，不是 SXS 物质数据。该模型也不包含 GRMHD、体吸收、偏振或自洽光谱辐射转移。随着 Roche-lobe 截断将各盘外缘压向 ISCO，稳定环带消失时发射通过 C² 过渡关闭；共同视界后严格为零，合并后发射未建模。该场景仍是<strong>强场 fast-light 近似，而非完整 NR</strong>。WebGL2 仅为弱场兼容预览，不声明双盘物理等价性。",
  "reference.schwarzschildTitle": "Schwarzschild 离线校准",
  "reference.kerrTitle": "Kerr 余留体离线校准",
  "reference.mode.composite": "合成图",
  "reference.mode.outcome": "光线结果",
  "reference.mode.lookback": "坐标回溯时间",
  "reference.mode.frequency": "频移因子 g",
  "reference.mode.null": "零性残差",
  "reference.mode.error": "投影误差",
  "reference.progress.manifestVerified": "{title} manifest SHA-256 已验证",
  "reference.progress.manifestFetching": "正在获取 {title} 的固定版本 manifest…",
  "reference.progress.sidecarVerified": "Manifest sidecar 已交叉验证",
  "reference.progress.sidecarChecking": "正在核对 manifest sidecar…",
  "reference.progress.chunks": "正在验证 transfer-map chunks {completed}/{total}…",
  "reference.progress.decoded": "已严格解码 {total} 条光线记录",
  "reference.progress.default": "正在验证 stationary transfer map…",
  "reference.outcomes": "{captured}% 捕获 · {escaped}% 逃逸",
  "reference.loadFailed": "Transfer map 验证失败：{message}",
  "reference.retry": "重试验证",
  "reference.returnBinary": "返回实时双黑洞",
  "reference.panelLabel": "显示设置",
  "reference.documentTitle": "{title} · 深空观测台",
  "reference.canvasAria": "{title}固定相机离线校准画面；点击像素或使用方向键检查光线记录",
  "reference.eyebrow": "研究工具 · 固定相机离线校准",
  "reference.fixedObserver": "固定观测相机",
  "reference.spin": "无量纲自旋",
  "reference.rayOutcome": "光线结果",
  "reference.massScale": "质量标度",
  "reference.displaySettings": "显示设置",
  "reference.fixedData": "固定数据",
  "reference.status.analytic": "解析真空参考",
  "reference.status.fixedCamera": "固定相机",
  "reference.status.chunks": "{verified}/{total} 数据块 SHA-256 已验证",
  "reference.modeGroup": "固定相机离线校准诊断视图",
  "reference.desktopHint": "点击查看光线记录 · 方向键移动 · Shift 加速",
  "reference.touchHint": "轻点画面查看像素光线记录",
  "reference.description.kerr": "解析 Kerr 余留体真空时空",
  "reference.description.schwarzschild": "解析 Schwarzschild 真空时空",
  "reference.physics": "{description}的固定相机离线校准画面，只用于研究与验证 transfer-map 数据链；不是双黑洞合并画面、NR 光追或高保真成品，不包含吸积发射。坐标：{coordinates}。积分器：{integrator}。",
  "reference.vacuum": "真空 · 无发射模型",
  "reference.static": "静态",
  "reference.fixedTetrad": "固定正交标架",
  "reference.pixel": "像素 x {x} · y {y}",
  "reference.loadingEyebrow": "固定信任根 · 正在验证",
  "reference.outcome.captured": "捕获（captured）",
  "reference.outcome.escaped": "逃逸（escaped）",
  "reference.outcome.unresolved": "未收敛（unresolved）",
  "reference.outcome.outsideDomain": "超出定义域（outside-domain）",
  "reference.outcome.integratorFailure": "积分器失败（integrator-failure）",
  "reference.outcome.missing": "缺失（missing）",
  "reference.outcome.unusable": "不可用（unusable）",
  "reference.none": "无",
  "renderer.fallback.deviceLost": "WebGPU 设备连接丢失",
  "renderer.fallback.renderError": "WebGPU 渲染异常",
  "renderer.webglMetalFallback": "WebGL2 · Metal 兼容回退",
  "renderer.appleGpu": "Apple GPU",
  "renderer.hardwareGpu": "硬件 GPU",
  "renderer.appleGpuMetal": "Apple GPU · Metal",
  "renderer.highPerformanceGpu": "高性能 GPU",
  "renderer.srgb": "sRGB 标准动态范围",
  "renderer.skyPending": "银河背景待载入",
  "renderer.p3ExtendedSdr": "P3 扩展 · 屏幕 SDR",
  "renderer.hdrDescription": "16 位浮点 Display‑P3 扩展 HDR（高光最高 4× SDR 白）",
  "renderer.p3Sdr": "Display‑P3 标准动态范围",
  "renderer.skyDetail": "{width}×{height} 原始全景 · 解析恒星层",
  "renderer.webglSdr": "WebGL2 回退路径使用 sRGB 标准动态范围",
  "renderer.webglSdrIntermediate": "WebGL2 回退路径使用 sRGB 标准动态范围 · {intermediate} 中间缓冲",
});

const CATALOGS = Object.freeze({ en: EN, "zh-CN": ZH_CN });

export function normalizeLocale(value) {
  return /^(zh|zh-cn|zh-hans)$/i.test(String(value || ""))
    ? "zh-CN"
    : DEFAULT_LOCALE;
}

export function isSupportedLocale(value) {
  return SUPPORTED_LOCALES.includes(value);
}

function languageParameter(value) {
  if (value?.get instanceof Function) {
    return value.has("lang") ? value.get("lang") : null;
  }
  const text = String(value || "");
  if (!text.includes("=") && !text.startsWith("?")) {
    return text || null;
  }
  const search = text.startsWith("?") ? text.slice(1) : text;
  const parameters = new URLSearchParams(search);
  return parameters.has("lang") ? parameters.get("lang") : null;
}

function storedLocale(storage) {
  try {
    const value = storage?.getItem?.("blackhole.language");
    return SUPPORTED_LOCALES.includes(value) ? value : null;
  } catch {
    return null;
  }
}

function defaultStorage() {
  try {
    return globalThis.localStorage;
  } catch {
    return null;
  }
}

export function localeFrom(
  value = globalThis.location?.search || "",
  storage = defaultStorage(),
) {
  const requested = languageParameter(value);
  return requested === null
    ? storedLocale(storage) ?? DEFAULT_LOCALE
    : normalizeLocale(requested);
}

function interpolate(template, values) {
  return template.replace(/\{([A-Za-z][A-Za-z0-9]*)\}/g, (match, name) => {
    if (!Object.hasOwn(values, name)) {
      throw new RangeError(`Missing i18n value ${name} for ${JSON.stringify(template)}`);
    }
    return String(values[name]);
  });
}

export function createI18n(
  value = globalThis.location?.search || "",
  storage = defaultStorage(),
) {
  const locale = localeFrom(value, storage);
  const catalog = CATALOGS[locale];
  return Object.freeze({
    locale,
    numberLocale: locale === "zh-CN" ? "zh-CN" : "en-US",
    t(key, values = {}) {
      const template = catalog[key] ?? EN[key];
      if (template === undefined) {
        throw new RangeError(`Unknown i18n key ${JSON.stringify(key)}`);
      }
      return interpolate(template, values);
    },
    formatNumber(valueToFormat, options) {
      return Number(valueToFormat).toLocaleString(
        locale === "zh-CN" ? "zh-CN" : "en-US",
        options,
      );
    },
  });
}

export function persistLocale(value, storage = defaultStorage()) {
  const locale = normalizeLocale(value);
  try {
    storage?.setItem?.("blackhole.language", locale);
  } catch {
    // Storage is an optional convenience. The URL still carries the choice.
  }
  return locale;
}

export function languageUrl(href, value) {
  const locale = normalizeLocale(value);
  const url = new URL(href);
  if (locale === DEFAULT_LOCALE) {
    url.searchParams.delete("lang");
  } else {
    url.searchParams.set("lang", locale);
  }
  return url.href;
}

export function applyDocumentI18n(documentRef, i18n = createI18n()) {
  documentRef.documentElement.lang = i18n.locale;
  for (const element of documentRef.querySelectorAll("[data-i18n]")) {
    element.textContent = i18n.t(element.dataset.i18n);
  }
  for (const element of documentRef.querySelectorAll("[data-i18n-html]")) {
    element.innerHTML = i18n.t(element.dataset.i18nHtml);
  }
  for (const attribute of ["aria-label", "title", "content"]) {
    const datasetKey = attribute.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    for (const element of documentRef.querySelectorAll(`[data-i18n-${attribute}]`)) {
      element.setAttribute(attribute, i18n.t(element.dataset[`i18n${datasetKey[0].toUpperCase()}${datasetKey.slice(1)}`]));
    }
  }
  const languageSelect = documentRef.getElementById("languageSelect");
  if (languageSelect) {
    languageSelect.value = i18n.locale;
  }
}

export function catalogKeys(locale = DEFAULT_LOCALE) {
  return Object.keys(CATALOGS[normalizeLocale(locale)]).sort();
}
