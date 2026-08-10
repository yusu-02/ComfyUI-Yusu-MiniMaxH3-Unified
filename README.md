线上工作流：http://runninghub.cn/post/2086063010979864577/?inviteCode=rh-v1525
       https://www.runninghub.ai/zh-cn/post/2086007179703197697/?inviteCode=rh-v1525
教程在B站/Youtube:在下鱼酥


# ComfyUI-Yusu-MiniMaxH3-Unified

统一 MiniMax H3 的文生视频、首尾帧生视频与全模态参考视频工作流，并提供节点内图片、视频、音频上传与精确裁剪。

> 本插件基于 ComfyUI 官方 `MiniMaxH3ImageToVideo` 与 `MiniMaxH3ReferenceToVideo` 节点进行封装，只负责输入组织、媒体处理和接口映射，不修改 MiniMax H3 的推理核心。

## 2026-08-10 更新

- 上传使用本地 ComfyUI 标准 `/upload/image` 资产入口。
- 波形画布不再把点击事件交给裁剪时间轴，修复点击“加载波形”反而移动裁剪标尺的问题。
- 当宿主未展开官方 Autogrow 媒体组时，仅将首个子接口恢复为可连接的 `IMAGE` / `AUDIO` 接口；本地已展开接口保持不变。
- 媒体面板宽度随节点宽度同步，避免宿主按旧宽度挂载后内容挤在左侧。
- 每段参考音频独立限制为 2–15 秒，多段音频总时长不再合并限制为 15 秒。
- 更新插件后必须完整重启 ComfyUI；若前后端版本错位，上传时会明确提示重启。

## 2026-08-09 更新

- 使用本地 ComfyUI 官方媒体路径解析。
- 自动音频时长模式保留有效的备用 `duration`。
- 裁剪时间轴增加画布事件隔离与指针捕获。
- 删除未调用的后端函数、单调用包装及旧版裁剪样式，保持现有功能不变。

## 功能概览

- 一个节点支持三种模式：
  - `text_to_video`：文生视频
  - `first_last_frame`：首尾帧生视频
  - `omni_reference`：图片、视频、音频全模态参考
- `clip`、`vae`、`audio_vae` 三个输入始终显示。
- 首尾帧和各类参考媒体接口根据模式动态显示。
- 节点内直接上传、预览、替换和裁剪媒体。
- 最多支持：
  - 9 张参考图片
  - 3 段参考视频
  - 3 段视频配对音频
  - 3 段独立参考音频
- 支持外部 ComfyUI `IMAGE` / `AUDIO` 接口覆盖节点内文件。
- 支持按参考音频自动计算生成时长。
- `duration` 使用秒数输入，并自动换算、对齐 H3 时间网格。
- 提供标准 `AUDIO` 输出，便于预览或保存实际使用的参考音频。
- 图片卡片使用统一标题、预览和按钮高度，横图与竖图保持对齐。
- 无遥测、无外部 API、无后台轮询、无空闲 GPU 任务。

## 运行要求

- 已安装并可正常运行的 ComfyUI。
- ComfyUI 版本中需要包含 MiniMax H3 官方节点：
  - `MiniMaxH3ImageToVideo`
  - `MiniMaxH3ReferenceToVideo`
- 已准备对应的 MiniMax H3 模型、CLIP、VAE；使用参考音频时还需要 Audio VAE。
- 插件没有独立的 `requirements.txt`，使用 ComfyUI 环境中的 PyTorch、Pillow、PyAV、NumPy 和 aiohttp。
- 系统安装 `ffprobe` 时会优先用于媒体探测；不可用时回退到 PyAV。

## 安装

### 方法一：Git 克隆

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yusu-02/ComfyUI-Yusu-MiniMaxH3-Unified.git
```

### 方法二：手动安装

1. 下载仓库 ZIP。
2. 解压后确认目录结构为：

```text
ComfyUI/
└── custom_nodes/
    └── ComfyUI-Yusu-MiniMaxH3-Unified/
        ├── __init__.py
        ├── nodes.py
        ├── media.py
        └── web/
```

3. 重启 ComfyUI。
4. 浏览器按 `Ctrl+F5` 强制刷新前端缓存。
5. 在节点菜单中查找：

```text
Yusu / MiniMax / H3 / Yusu MiniMax H3 Unified
```

升级旧版本时，建议先完整删除旧插件目录，再复制新版本，避免残留旧 JavaScript 或 Python 文件。

## 重要：模型连接方式

本节点**没有 `MODEL` 输入或输出**。

模型加载器应直接连接到后续官方采样链，例如基础引导、调度器或采样器；本节点只输出 conditioning 和 AV latent。

```text
模型加载器 ───────────────→ Guider / Scheduler / Sampler
CLIP ───────→ Unified.clip
VAE ────────→ Unified.vae
Unified.positive ─────────→ Guider.conditioning
Unified.av_latent ────────→ Sampler.latent_image
```

模式与模型家族应正确匹配：

- `text_to_video`、`first_last_frame`：使用对应的 FL2VA 模型。
- `omni_reference`：使用对应的 Ref2VA 模型。

## 节点接口

### 始终显示的输入

| 接口 | 类型 | 说明 |
|---|---|---|
| `clip` | `CLIP` | 用于 prompt 编码 |
| `vae` | `VAE` | 用于图像/视频 latent 处理 |
| `audio_vae` | `VAE` | 始终显示；只有存在参考音频时才必须连接 |
| `prompt` | `STRING` | 支持外部文本节点和动态 prompt |
| `width` | `INT` | 输出宽度，必须是 32 的倍数 |
| `height` | `INT` | 输出高度，必须是 32 的倍数 |
| `duration` | `FLOAT` | 目标生成秒数，允许 0，不设插件上限 |
| `有音频时自动长度` | `BOOLEAN` | 参考模式下按最长有效音频自动计算时长 |

### 按模式显示的输入

#### `text_to_video`

不增加媒体接口，只使用 prompt、宽高和 `duration`。

#### `first_last_frame`

| 接口 | 类型 | 说明 |
|---|---|---|
| `first_frame` | `IMAGE` | 可选首帧，外部接口优先于节点内上传 |
| `last_frame` | `IMAGE` | 可选尾帧，外部接口优先于节点内上传 |

首帧和尾帧至少需要一个。

#### `omni_reference`

| 接口 | 类型 | 数量 | 说明 |
|---|---|---:|---|
| `ref_image_0...` | `IMAGE` | 9 | 参考图片 |
| `ref_video_0...` | `IMAGE` 批次 | 3 | 24 FPS 参考视频帧批次 |
| `ref_video_audio_0...` | `AUDIO` | 3 | 与同编号参考视频配对的音轨 |
| `ref_audio_0...` | `AUDIO` | 3 | 独立参考音频 |
| `ref_image_size` | `COMBO` | 1 | `match` 或 `max` |

### 输出

| 接口 | 类型 | 说明 |
|---|---|---|
| `positive` | `CONDITIONING` | 连接到引导节点的条件输入 |
| `av_latent` | `LATENT` | 联合视频/音频 latent，连接到采样链 |
| `audio` | `AUDIO` | 输出实际参与参考的最长一段裁剪后音频；无音频时为空 |

`audio` 输出的是参考音频，不是模型生成后的最终视频音轨。

## 三种模式

### 1. 文生视频

选择：

```text
mode = text_to_video
```

只需要：

- `clip`
- `vae`
- prompt
- 宽度、高度
- `duration`

节点内上传的参考素材会被忽略。

### 2. 首尾帧生视频

选择：

```text
mode = first_last_frame
```

提供首帧、尾帧或两者。可以：

- 连接外部 `IMAGE`；
- 在节点面板内上传图片。

同一个槽位同时存在外部连接和节点内图片时，外部连接优先。

### 3. 全模态参考视频

选择：

```text
mode = omni_reference
```

可组合使用：

- 参考图片
- 参考视频
- 视频内嵌原声
- 视频配对音频
- 独立参考音频

存在任意参考音频时必须连接 `audio_vae`。

## 节点内媒体与外部接口映射

节点内槽位从 1 开始显示，ComfyUI 外部接口从 0 开始编号：

```text
节点内参考图 1       ↔ ref_image_0
节点内参考视频 1     ↔ ref_video_0
视频 1 配对音频      ↔ ref_video_audio_0
节点内独立音频 1     ↔ ref_audio_0
```

外部接口会覆盖同槽位的节点内文件，不会重复加入两份参考。

视频音轨优先级：

```text
外部 ref_video_audio_N
        ↓
节点内“视频 N 配对音频”
        ↓
参考视频的内嵌原声
```

`ref_video_audio_N` 必须与同编号 `ref_video_N` 一起使用。孤立的配对音频会在解码前报错，避免无意义的 CPU 和内存消耗。

## Prompt 标签

全模态参考模式会在节点面板中显示实际标签映射，例如：

```text
<Picture 1> ← ref_image_0
<Audio 1>   ← ref_audio_0
<Video 1>   ← ref_video_0
```

prompt 必须显式使用需要参考的标签，例如：

```text
使用 <Audio 1> 的音色和说话方式，
让 <Picture 1> 中的人物自然说话。
```

注意：

- 外部接口编号从 0 开始，prompt 标签从 1 开始。
- 视频配对音频会参与标签排序。
- 不要根据接口后缀猜测 `<Audio n>`，应以节点面板中的映射为准。
- prompt 来自外部文本节点时，插件会尽量读取上游文本；动态生成且前端无法读取时，会交由后端按最终 prompt 执行，不会直接误报缺失。

## 时长计算

### 手动时长

`duration` 的单位是秒，接口显示为：

```text
FLOAT (duration)
```

计算流程：

```text
目标秒数
→ round(秒数 × 24 FPS)
→ 对齐到 H3 的 17k+5 帧网格
→ 创建 AV latent
```

例如：

```text
5.000 秒
→ 120 帧
→ 对齐为 124 帧
→ 实际约 5.167 秒
```

输入 `0` 秒不会产生真正的零帧 latent，而会得到 H3 时间结构允许的最小 5 帧，约 `0.208` 秒。

插件不限制最大输出时长。超长生成会线性增加主存、显存、采样时间和输出文件体积，数值过大可能导致内存不足或驱动崩溃。

### 按音频自动长度

仅在 `omni_reference` 模式生效。

启用“有音频时自动长度”后：

- `duration` 控件变灰并锁定；
- 使用裁剪后最长的一段有效参考音频；
- 自动按 24 FPS 换算并对齐 H3 帧网格；
- 没有有效参考音频时回退到已保存的手动秒数。

文生视频、首尾帧生视频和普通无声参考不会要求额外加载音频时长节点。

## 参考媒体限制

插件对参考素材执行以下校验：

- 每段参考视频：2–15 秒。
- 所有参考视频总时长：不超过 15 秒。
- 每段参考音频：2–15 秒。
- 多段参考音频按单段分别校验，不限制合计时长。
- 前端裁剪标尺允许任意选择；执行时才校验当前单段选区时长，超出范围会报错。
- 生成宽高必须是 32 的倍数。
- 单个上传文件最大 512 MiB。
- 图片最大 100,000,000 像素。
- 参考音频采样率最高 384 kHz。

这些是参考媒体限制，与不设上限的输出 `duration` 是两套不同规则。

## 音频处理

- 裁剪区间使用半开区间 `[入点, 出点)`。
- 单声道自动复制为双声道。
- 多声道只保留前两个声道。
- 音频波形限制在 `[-1, 1]`。
- 空音频、纯静音、NaN、无穷值会直接报错。
- 节点内长音频只解码裁剪区间，不先加载完整文件。
- 视频和音频起始时间不一致时，会按视频时间轴校正，并在需要时补静音。
- 前端波形默认不自动解码；点击波形区域后才加载。

## 视频处理

- 视频解码后统一转换为 packed RGB24。
- 应用旋转元数据和非方形像素比例。
- 在线采样为 24 FPS。
- 解码阶段直接适配 H3 参考画布。
- 只物化 conditioning 实际需要的帧前缀，减少高帧率、长视频的内存压力。

## 上传文件位置

本地 ComfyUI 默认将节点内上传的文件保存在：

```text
ComfyUI/input/minimax_h3_unified/
```

节点中的“删除”按钮只移除当前槽位引用，不删除磁盘文件，以免破坏其他工作流对同一素材的引用。不再使用的文件需要手动清理。

上传过程会检查：

- 扩展名
- MIME 类型
- 图片真实格式
- 视频/音频流
- 文件大小
- 路径安全

解析失败或内容无效时不会写入槽位状态。

## 性能设计

- prompt 变化只刷新音频标签状态，不重建整个媒体面板。
- 裁剪滑块的状态保存合并到浏览器动画帧。
- 图片使用延迟加载和异步解码。
- 视频、音频预览只预加载元数据。
- 波形缓存只保存降采样峰值，不保存完整 `AudioBuffer`。
- 波形缓存最多 12 项。
- 大于 64 MiB 或超过 5 分钟的音频不会在浏览器内整段解码波形。
- 媒体探测在线程池执行，不阻塞 ComfyUI Web 服务。
- 没有后台轮询、常驻工作线程或空闲 GPU 运算。

## 模式切换说明

`clip`、`vae`、`audio_vae` 始终保留。

以下接口按模式动态创建或删除：

- `first_frame`
- `last_frame`
- `ref_image_*`
- `ref_video_*`
- `ref_video_audio_*`
- `ref_audio_*`

ComfyUI 删除动态接口时可能同时断开该接口的外部连线。因此切换模式后，动态媒体接口可能需要重新连接；节点内上传的媒体状态仍会保存在工作流中。

## 常见问题

### 已连接音频，为什么提示没有引用？

确认以下条件：

1. 模式为 `omni_reference`；
2. 已连接 `audio_vae`；
3. 至少存在一张参考图片或一段参考视频；
4. prompt 包含面板显示的 `<Audio n>` 标签；
5. 参考音频裁剪后不是静音，且时长符合限制。

### 为什么输入秒数和实际生成时长不完全一致？

H3 时间维度必须满足 `17k+5`，因此节点会向上对齐帧数。面板会显示目标秒数、有效帧数和实际秒数。

### 为什么没有 `MODEL` 接口？

模型不需要经过本节点中转。将模型加载器直接连接到 Guider、Scheduler 或 Sampler，本节点只提供 `positive` 和 `av_latent`。

### 为什么节点中的“删除”没有删除硬盘文件？

同一文件可能被其他工作流引用。为避免破坏其他工作流，插件只删除槽位引用，物理文件需要手动清理。

### 为什么音频输出不是最终生成音轨？

`audio` 输出用于返回实际送入参考编码的音频。最终生成音轨需要从采样和解码后的 AV latent 工作流中取得。

### 更换插件后界面仍然是旧版本怎么办？

1. 确认旧插件目录已完整删除；
2. 重启 ComfyUI 后端；
3. 浏览器按 `Ctrl+F5`；
4. 必要时清除浏览器站点缓存。

## 代码结构

```text
ComfyUI-Yusu-MiniMaxH3-Unified/
├── __init__.py
├── media.py
├── nodes.py
├── web/
│   ├── minimax_h3_unified.js
│   └── minimax_h3_unified.css
└── tests/
    ├── test_plugin.py
    └── test_frontend.mjs
```

- `__init__.py`：插件入口和上传路由。
- `media.py`：路径安全、媒体探测、裁剪、解码与校验。
- `nodes.py`：节点 schema、模式路由、槽位映射与官方节点调用。
- `web/`：节点面板、上传交互、动态接口和样式。
- `tests/`：后端和前端回归测试，不会随 ComfyUI 自动运行。

## 开发与测试

后端测试：

```bash
python -m unittest discover \
  -s custom_nodes/ComfyUI-Yusu-MiniMaxH3-Unified/tests \
  -v
```

前端测试：

```bash
node custom_nodes/ComfyUI-Yusu-MiniMaxH3-Unified/tests/test_frontend.mjs
```

语法检查：

```bash
python -m py_compile \
  custom_nodes/ComfyUI-Yusu-MiniMaxH3-Unified/__init__.py \
  custom_nodes/ComfyUI-Yusu-MiniMaxH3-Unified/media.py \
  custom_nodes/ComfyUI-Yusu-MiniMaxH3-Unified/nodes.py

node --check \
  custom_nodes/ComfyUI-Yusu-MiniMaxH3-Unified/web/minimax_h3_unified.js
```

## 已知限制

- 插件依赖 ComfyUI 当前 MiniMax H3 官方节点的接口，官方接口发生不兼容变更时可能需要同步更新。
- 动态模式接口在切换模式时可能断开外部连线。
- 高分辨率、长参考视频最终仍会形成较大的 `IMAGE` 批次，占用较多主存。
- 超长输出没有插件上限，但不代表模型已针对所有长度训练或验证。
- 参考音频进入 conditioning 不代表模型一定逐字复现内容或完全复制音色。
- 损坏、缺少时间戳或编码异常的媒体，PyAV/FFmpeg 可能无法准确解码。

## 安全与隐私

- 不收集遥测数据。
- 不向外部服务器发送媒体、prompt 或工作流。
- 前端网络请求仅访问当前本地 ComfyUI 的标准上传、插件媒体检查和 `/view` 路由。
- 不包含挖矿、后台监控、常驻轮询或自动执行测试代码。

---

欢迎通过 Issue 提交可复现的问题。建议同时附上：

- ComfyUI 版本或提交号
- 操作系统
- Python、PyTorch、PyAV 版本
- 使用模式
- 完整错误日志
- 可复现的最小工作流
