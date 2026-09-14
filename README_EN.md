# BSAI-ComfyUI-TaoMate

[**中文说明 (Chinese)**](./README.md) · **English**

Wrap the open-source **TaoMate-H3** (by Alibaba Taobao Live team) — a low-latency
streaming audio-video generation runtime built on **MiniMax H3**, featuring
**3-step distilled LoRA + KV-cache streaming continuation**
("Stream. Speak. Continue.") — into **native ComfyUI nodes and runnable workflows**.

The reference video (BV1SNYq6gE8A) explicitly states that TaoMate-H3 is a
**standalone scheduling runtime and currently has no ComfyUI support**. This
plugin fills that gap: load the TaoMate 3-step LoRA in your local ComfyUI, run
3-step generation, and produce long audio-video (digital-human style) with
"last-frame + tail-audio anchor" streaming continuation in one click.

---

## 1. Feasibility (video / paper / LoRA verified)

| Item | Conclusion |
| --- | --- |
| Video | TaoMate-H3 = 3-step distilled LoRA + KV-cache streaming continuation; audio-video sync, long videos, digital human; official multi-GPU Hopper pure-DiT is 11.45× faster |
| Paper | TaoMate (arXiv 2607.24359, Taobao Live + Nanjing Univ.): 22.1B, 3 denoising steps per block, anchor-guided persistent memory, ~35 FPS multi-GPU stage-parallel, ~11 FPS single device |
| LoRA verified | `TaoMate-H3-step3000-ComfyUI-FP32.safetensors`: **624 tensors = 50 DiT blocks (attn.qkv_proj / out_proj / mlp.fc1 / fc2) + token_refiner.blocks.0-1**, rank 128 / alpha 128 / 3000 steps, base model marked `MiniMax-H3 FL2VA` |
| Key-name match | Matches ComfyUI native MiniMax-H3 FL2VA **tensor-by-tensor** (incl. token_refiner), **no adaln / memory-attention / FiLM weights** → loads via standard `comfy.sd.load_lora_for_models`, no architecture changes |
| Conclusion | ✅ **Feasible.** 3-step distillation + chunked continuation works on a single GPU ComfyUI (verified with all required models on a local RTX 5090 24GB) |

> ⚠️ Scope: the official 35 FPS multi-GPU stage-parallel pipeline depends on its
> standalone runtime (3–8 Hopper cards). This plugin does **not** replicate that
> multi-GPU pipeline; the single-GPU path is **chunk-by-chunk serial 3-step
> generation**, approximating official persistent memory with
> "previous chunk's last frame (first-frame anchor) + previous chunk's tail audio
> (frame-0 audio anchor)" to keep appearance and voice consistent across long
> videos. The official LoRA contains only 3-step distillation weights; memory
> attention and other architectural modules are not in the LoRA and cannot be
> reproduced on the native H3 model.

---

## 2. Installation

```powershell
# 1) Put this folder into custom_nodes:
#    ComfyUI/custom_nodes/BSAI-ComfyUI-TaoMate/
# 2) Make sure the TaoMate LoRA is in your loras folder (FP32 or BF16, auto-listed):
#    ComfyUI/models/loras/TaoMate-H3-step3000-ComfyUI-FP32.safetensors
#    ComfyUI/models/loras/TaoMate-H3-step3000-ComfyUI-BF16.safetensors
# 3) Restart ComfyUI
```

**Alternative (ComfyUI Manager):** Manager → Install Custom Nodes → paste
`https://github.com/xm6018924/BSAI-ComfyUI-TaoMate` → install → restart.

Dependencies: `torch` / `torchaudio` / `tqdm` (already present in a ComfyUI env).

### Required models (MiniMax-H3 base workflow)

- Diffusion model: `models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`
  (19.5GB, default for a 24GB card; use `minimax_h3_fl2va_bf16.safetensors` 61.7GB for max precision)
- Text encoder: `models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (CLIPLoader type=`minimax`)
- Video VAE: `models/vae/minimax_h3_video_vae_fp16.safetensors`
- Audio VAE: `models/vae/minimax_h3_audio_vae_fp32.safetensors`

---

## 3. Nodes (category: `BSAI/TaoMate`)

| Node | Input → Output | Purpose |
| --- | --- | --- |
| **BSAI TaoMate LoRA Loader** | MODEL → MODEL | Loads the TaoMate-H3 3-step LoRA (auto-lists TaoMate files in `loras`; strength default 1.0; strict filename check; prints key-match stats) |
| **BSAI TaoMate 3-Step Timesteps** | MODEL + ladder → SIGMAS | Converts the explicit training ladder to SIGMAS, default `999,750,500` (3 steps, auto-appends 0) |
| **BSAI TaoMate Euler Sampler** | → SAMPLER | 3-step Euler: single schedule on new ComfyUI (ModelSamplingAV); auto-falls back to legacy dual schedule (video shift 12 / audio shift 3) |
| **BSAI TaoMate Continuation Probe** | LATENT+VAE+AudioVAE → IMAGE+AUDIO+INT | Extracts the previous chunk's last frame + tail audio for the next chunk |
| **BSAI TaoMate Stream Chain** | MODEL+CLIP+VAE+AudioVAE+SIGMAS+prompts → IMAGE+AUDIO+last-frame+tail-audio+chunk-count | **One-click streaming long video**: one prompt per line = one chunk, 3-step generation per chunk with auto continuation, outputs concatenated audio-video |

### Recommended graph (3-step single block)

```
UNETLoader(MiniMax H3 FL2VA)
  → BSAI TaoMate LoRA Loader
  → MiniMaxH3SigmaShift(shift_video 12, shift_audio 3)
  → BSAI TaoMate 3-Step Timesteps ───────────────┐
  → BSAI TaoMate Euler Sampler ──────────────────┤
CLIPLoader(minimax) ──┐                          │
VAELoader(video) ──┐  │                          │
VAELoader(audio) ──┼─> MiniMaxH3ImageToVideo ────┤
                  │   (positive → BasicGuider)   │
                  └── (LATENT → RandomNoise)     │
SamplerCustomAdvanced(noise+guider+sampler+sigmas+latent_image)
  → VAEDecode(video) / VAEDecodeAudio(audio) → CreateVideo(fps 24) → SaveVideo
```

### One-click streaming long video (digital-human style)

```
(same LoRA / SigmaShift / Timesteps / Euler setup as above)
BSAITaoMateStreamChain(
    model, clip, vae, audio_vae, sigmas,
    prompts=one prompt per line,  # each line = one chunk, auto-continued
    width/height/length, seed,
    anchor_tail_audio=True, tail_audio_seconds=1.5)
→ IMAGE(video) + AUDIO → CreateVideo(fps 24) → SaveVideo
(optional first_frame start anchor; outputs last-frame/tail-audio can feed another StreamChain)
```

> The 3-step distilled model does **not** use CFG (cfg fixed at 1.0) and needs no negative prompt.

---

## 4. Example workflows

Under `example_workflows/`:

- `BSAI_TaoMate_3step_T2VA.json` — 3-step single-block text-to-audio-video (with audio)
- `BSAI_TaoMate_Stream_DigitalHuman.json` — streaming digital-human long video (multi-chunk auto continuation)

Load into ComfyUI, check the model paths, fill in the prompt, and queue.

---

## 5. Known limitations

1. **Single-GPU serial**: not the official multi-GPU 35 FPS real-time streaming;
   chunk-by-chunk 3-step generation, roughly tens of seconds per chunk on a 5090.
2. **Chunk seam**: approximates official persistent memory with last-frame +
   tail-audio anchors; appearance/voice stay consistent, but seam frames and audio
   waveforms are not bit-continuous — fast camera moves may show slight jumps
   (mitigate with larger `tail_audio_seconds`, or prompt the next chunk with
   "continue the previous segment, keep motion coherent").
3. **LoRA has no memory attention/FiLM**: the full official memory mechanism
   cannot be reproduced on the native H3 model.
4. **int8 base + FP32 LoRA**: quantization-aware patching is handled by ComfyUI;
   if quality is off, switch to `minimax_h3_fl2va_bf16.safetensors`.
5. The ladder default `999,750,500` is a recommended start for 3-step
   distillation; tune per output.

---

## 6. References

- Video: BV1SNYq6gE8A (阿里开源黑科技！MiniMaxH3直接变流式，音画同步长视频速度暴涨十一倍)
- Project page: https://taoliveaigc.github.io/TaoMate
- Paper: TaoMate: Anchor-Guided Memory Bridging Evolving and Reference States for
  Real-Time Audio-Video Digital Human Generation (arXiv 2607.24359)
- Upstream repo: github.com/TaoMateAI/TaoMate-H3 (runtime + LoRA weights)
- LoRA conversion: TaoMate-H3-step3000 → ComfyUI format (lossless key rename, per-module alpha, FP32/BF16)
