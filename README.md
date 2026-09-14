# BSAI-ComfyUI-TaoMate

**中文说明** · [**English (README_EN)**](./README_EN.md)

把 **阿里淘宝直播团队开源的 TaoMate-H3**（基于 MiniMax H3 的低延迟流式音视频生成：
**3 步蒸馏 LoRA + KV 缓存流式续写**，主打 “Stream. Speak. Continue.”）封装为
**ComfyUI 原生节点与可运行工作流**。

视频（BV1SNYq6gE8A）中明确说明：TaoMate-H3 是一个**独立的调度运行时**，
**不是 ComfyUI 自定义节点、目前也没有 ComfyUI 支持**。本插件就是补上这一环：
在你的本地 ComfyUI 里直接加载 TaoMate 3 步 LoRA、跑 3 步生成、并以
「末帧 + 尾音频锚定」的流式续写方式一键生成长音视频（数字人风格）。

---

## 1. 可行性结论（视频/论文/LoRA 实测）

| 项目 | 结论 |
| --- | --- |
| 视频内容 | TaoMate-H3 = 3 步蒸馏 LoRA + KV 缓存流式续写；音画同步、长视频、数字人；官方多卡 Hopper 纯 DiT 快 11.45 倍 |
| 论文 | TaoMate（arXiv 2607.24359，淘宝直播 + 南大）：22.1B、每块 3 步去噪、锚点引导持久记忆、多卡阶段并行 35 FPS、单设备约 11 FPS |
| LoRA 实测 | `TaoMate-H3-step3000-ComfyUI-FP32.safetensors`：**624 张量 = 50 个 DiT block（attn.qkv_proj/out_proj/mlp.fc1/fc2）+ token_refiner.blocks.0-1**，rank 128 / alpha 128 / 3000 步，底模标注 `MiniMax-H3 FL2VA` |
| 键名匹配 | 与 ComfyUI 原生 MiniMax-H3 FL2VA 模型**逐键匹配**（含 token_refiner），**不含 adaln / 记忆注意力 / FiLM 权重** → 走标准 `comfy.sd.load_lora_for_models` 即可，无需改模型架构 |
| 结论 | ✅ **可做**。3 步蒸馏 + 分块续写在单卡 ComfyUI 上完全可行（本机 RTX 5090 24GB 已验证配套模型齐全） |

> ⚠️ 边界说明：官方 35 FPS 多卡阶段并行依赖其独立运行时（3-8 卡 Hopper），
> 本插件不复制该多卡管线；单卡路径为**逐块串行 3 步生成**，以
> 「上一块末帧（首帧锚）+ 上一块尾音频（frame 0 音频锚）」近似官方持久记忆，
> 实现外观与声线连续的长视频。官方 LoRA 只含 3 步蒸馏权重，记忆注意力等
> 架构模块不在 LoRA 中，无法在原生 H3 模型上还原官方全部记忆机制。

---

## 2. 安装

```powershell
# 1) 把本文件夹放入 custom_nodes：
#    ComfyUI/custom_nodes/BSAI-ComfyUI-TaoMate/
# 2) 确认 loras 目录下有 TaoMate LoRA（二选一，FP32/BF16 均可，插件自动列出）：
#    ComfyUI/models/loras/TaoMate-H3-step3000-ComfyUI-FP32.safetensors
#    ComfyUI/models/loras/TaoMate-H3-step3000-ComfyUI-BF16.safetensors
# 3) 重启 ComfyUI
```

依赖：torch / torchaudio / tqdm（ComfyUI 环境自带）。

**也可通过 ComfyUI Manager 安装**：Manager → Install Custom Nodes → 粘贴
`https://github.com/xm6018924/BSAI-ComfyUI-TaoMate` → 安装 → 重启。

### 配套模型（MiniMax-H3 基础工作流所需）

- 扩散模型：`models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`
  （19.5GB，24GB 卡默认；追求极致精度可用 `minimax_h3_fl2va_bf16.safetensors` 61.7GB）
- 文本编码：`models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors`（CLIPLoader type=`minimax`）
- 视频 VAE：`models/vae/minimax_h3_video_vae_fp16.safetensors`
- 音频 VAE：`models/vae/minimax_h3_audio_vae_fp32.safetensors`

---

## 3. 节点说明（分类：`BSAI/TaoMate`）

| 节点 | 输入 → 输出 | 作用 |
| --- | --- | --- |
| **BSAI TaoMate LoRA Loader** | MODEL → MODEL | 加载 TaoMate-H3 3 步 LoRA（自动列出 loras 中 TaoMate 文件；strength 默认 1.0；strict 校验文件名；打印键匹配统计） |
| **BSAI TaoMate 3-Step Timesteps** | MODEL + ladder → SIGMAS | 显式训练阶梯换算 SIGMAS，默认 `999,750,500`（3 步，末尾自动补 0） |
| **BSAI TaoMate Euler Sampler** | → SAMPLER | 3 步 Euler：新版 ComfyUI（ModelSamplingAV）单调度；旧版自动退化音视频双调度（video shift 12 / audio shift 3） |
| **BSAI TaoMate Continuation Probe** | LATENT+VAE+AudioVAE → IMAGE+AUDIO+INT | 取上一块末帧 + 尾音频，供下一块续接 |
| **BSAI TaoMate Stream Chain** | MODEL+CLIP+VAE+AudioVAE+SIGMAS+prompts → IMAGE+AUDIO+末帧+尾音频+块数 | **一键流式长视频**：每行一个分块提示词，逐块 3 步生成并自动续接，输出拼接音视频 |

### 推荐链路（3 步单块）

```
UNETLoader(MiniMax H3 FL2VA)
  → BSAI TaoMate LoRA Loader
  → MiniMaxH3SigmaShift(shift_video 12, shift_audio 3)
  → BSAI TaoMate 3-Step Timesteps ───────────────┐
  → BSAI TaoMate Euler Sampler ──────────────────┤
CLIPLoader(minimax) ──┐                          │
VAELoader(视频) ──┐   │                          │
VAELoader(音频) ──┼─> MiniMaxH3ImageToVideo ─────┤
                  │   (positive → BasicGuider)   │
                  └── (LATENT → RandomNoise)     │
SamplerCustomAdvanced(noise+guider+sampler+sigmas+latent_image)
  → VAEDecode(视频) / VAEDecodeAudio(音频) → CreateVideo(fps 24) → SaveVideo
```

### 一键流式长视频（数字人风格）

```
（上方同款 LoRA / SigmaShift / Timesteps / Euler 准备）
BSAITaoMateStreamChain(
    model, clip, vae, audio_vae, sigmas,
    prompts=每行一段提示词,   # 每段=一个分块，自动续接
    width/height/length, seed,
    anchor_tail_audio=True, tail_audio_seconds=1.5)
→ IMAGE(video) + AUDIO → CreateVideo(fps 24) → SaveVideo
（可选 first_frame 起始锚图；输出末帧/尾音频可再接到下一个 StreamChain 继续）
```

> 3 步蒸馏模型 **不做 CFG**（cfg 固定 1.0），也不需要 negative prompt。

---

## 4. 示例工作流

`example_workflows/` 下：

- `BSAI_TaoMate_3step_T2VA.json` — 3 步单块文生音视频（含音频）
- `BSAI_TaoMate_Stream_DigitalHuman.json` — 流式数字人长视频（多块自动续接）

拖入 ComfyUI 后：核对模型文件路径 → 填提示词 → 队列运行。

---

## 5. 已知限制

1. **单卡串行**：非官方多卡 35 FPS 实时流式；逐块 3 步生成，5090 上单块约十几秒级。
2. **块间衔接**：以末帧+尾音频锚定近似官方持久记忆；画面/声线保持良好，但衔接帧
   与音频波形非逐位连续，长镜头剧烈运动时可能轻微跳变（可增大 tail_audio_seconds、
   或给下一块提示词补充“继续上一段，动作连贯”类描述缓解）。
3. **LoRA 不含记忆注意力/FiLM**：无法在原生 H3 模型上还原官方完整记忆机制。
4. **int8 底模 + FP32 LoRA**：量化感知补丁由 ComfyUI 处理；如遇质量异常，
   建议换 `minimax_h3_fl2va_bf16.safetensors` 底模。
5. 阶梯默认值 `999,750,500` 为 3 步蒸馏的推荐起点，可据画面微调。

---

## 6. 参考

- 视频：BV1SNYq6gE8A《阿里开源黑科技！MiniMaxH3直接变流式，音画同步长视频速度暴涨十一倍》
- 项目页：https://taoliveaigc.github.io/TaoMate
- 论文：TaoMate: Anchor-Guided Memory Bridging Evolving and Reference States for
  Real-Time Audio-Video Digital Human Generation（arXiv 2607.24359）
- 开源仓库：github.com/TaoMateAI/TaoMate-H3（运行时 + LoRA 权重）
- LoRA 转换：TaoMate-H3-step3000 → ComfyUI 格式（lossless 键重命名、逐模块 alpha、FP32/BF16）
