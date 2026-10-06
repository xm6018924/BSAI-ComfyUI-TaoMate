"""
BSAI-ComfyUI-TaoMate - nodes.py

围绕 TaoMate-H3（阿里淘宝直播团队开源的 MiniMax-H3 3 步蒸馏 + 流式续写
运行时）封装的 ComfyUI 节点：

  * BSAITaoMateLoRALoader        - TaoMate-H3 3 步蒸馏 LoRA 加载 (MODEL->MODEL)
  * BSAITaoMateTimesteps         - 3 步显式时间阶梯 (MODEL -> SIGMAS)
  * BSAITaoMateEulerSampler      - 音视频双调度 Euler 采样器 (-> SAMPLER)
  * BSAITaoMateContinuationProbe - 上一块末帧 + 尾音频探针 (LATENT -> IMAGE+AUDIO)
  * BSAITaoMateStreamChain       - 一键流式长视频 (多块连续生成自动续接)

设计说明
--------
- LoRA 键名与 ComfyUI 原生 MiniMax-H3 FL2VA 模型完全一致
  （blocks.0-49 的 attn.qkv_proj/out_proj/mlp.fc1/fc2 + token_refiner.blocks.0-1），
  不含 adaln / 记忆注意力 / FiLM 权重，因此走 ComfyUI 标准
  comfy.sd.load_lora_for_models 即可，无需特殊注入。
- 采样器与 BSAI-ComfyUI-FastH3 / ComfyUI-MiniMax-H3-Turbo 同构：
  新版 ComfyUI 用 ModelSamplingAV 原生处理音视频双调度时走单调度 Euler；
  旧版无 ModelSamplingAV 时退化为音视频各自 flow shift 的 legacy 双调度。
- 流式续写 = 分块生成 + 首帧锚定 + 尾音频锚定：下一块以"上一块末帧"作为
  first_frame 锚点、"上一块尾音频"作为 frame 0 音频锚点，实现外观与声线
  的连续；这是官方持久记忆运行时的 ComfyUI 可行近似（单卡即可跑）。
- 官方 TaoMate 多卡阶段并行（3-8 卡 Hopper、35 FPS）依赖其独立运行时，
  本插件不复制该多卡管线，单卡路径为逐块串行 3 步生成。
"""

import math
import os
import struct
import sys
import json

import torch
import torchaudio
from tqdm.auto import trange

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.sd
import comfy.lora
import node_helpers
import folder_paths

from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise

try:
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent, _resize
except Exception as _e:  # pragma: no cover
    raise ImportError(
        "[BSAI TaoMate] 需要带原生 MiniMax-H3 支持的 ComfyUI "
        "(comfy_extras.nodes_minimax_h3 缺失，请先更新 ComfyUI)。"
    ) from _e

# ---- BSAI 插件协同 SDK：加载即自动注册（失败不拖垮插件） ----
_ORCH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "BSAI-ComfyUI-Orchestrator")
if not os.path.isdir(_ORCH):
    _ORCH = r"G:\BSAI-ComfyUI-intel-XPU-GPU-NPU-aki\ComfyUI\custom_nodes\BSAI-ComfyUI-Orchestrator"
if os.path.isdir(_ORCH) and _ORCH not in sys.path:
    sys.path.insert(0, _ORCH)
try:
    from bsai_orch_client import BSAIOrch
except Exception:
    BSAIOrch = None

try:
    if BSAIOrch is not None:
        BSAIOrch.register(
            name="BSAI-TaoMate",
            kind="sampling",                 # 能力类型：GPU1 视频采样
            hardware=["cuda"],
        )
except Exception:
    pass

SHIFT_V, SHIFT_A = 12.0, 3.0   # MiniMax H3 视频/音频 flow shift


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _read_safetensors_metadata(path):
    """轻量读取 safetensors 头部 __metadata__，不加载权重。"""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 512 * 1024 * 1024:
                return {}
            hdr = json.loads(f.read(n))
        return hdr.get("__metadata__", {}) or {}
    except Exception:
        return {}


def _time_shift_sigma(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return to * base / (1.0 + (to - 1.0) * base)


def _time_shift_slope(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return (to * (1.0 + (fr - 1.0) * base) ** 2) / (fr * (1.0 + (to - 1.0) * base) ** 2)


def _audio_sigma(sv, shift_v, shift_a):
    return _time_shift_sigma(sv, shift_v, shift_a)


def _audio_slope(sv, shift_v, shift_a):
    return _time_shift_slope(sv, shift_v, shift_a)


def _latent_shapes(model):
    """[video_shape, audio_shape] - H3 平铺 latent 为 [video | audio] 拼接。"""
    guider = getattr(model, "inner_model", model)
    conds = getattr(guider, "conds", None)
    if conds:
        for cond_list in conds.values():
            for c in (cond_list or []):
                mc = c.get("model_conds", {}) if isinstance(c, dict) else {}
                if "latent_shapes" in mc:
                    return mc["latent_shapes"].cond
    return None


def _model_sampling(model):
    for chain in (("inner_model", "inner_model", "model_sampling"),
                  ("inner_model", "model_sampling"),
                  ("model_sampling",)):
        o = model
        try:
            for a in chain:
                o = getattr(o, a)
        except AttributeError:
            continue
        if o is not None:
            return o
    return None


def _native_av_schedule(model):
    """True 时 ComfyUI 用 ModelSamplingAV 原生处理 H3 音视频双调度，单调度 Euler 即可。"""
    ms = _model_sampling(model)
    if ms is None:
        return False
    if getattr(ms, "audio_shift", None) is not None:
        return True
    av = getattr(comfy.model_sampling, "ModelSamplingAV", None)
    return av is not None and isinstance(ms, av)


def make_taomate_euler(shift_video=SHIFT_V, shift_audio=SHIFT_A,
                       schedule_mode="auto", verbose=True):
    """生成 TaoMate 3 步 Euler 采样函数（绑定 shift / schedule / verbose）。"""
    @torch.no_grad()
    def _taomate_euler(model, x, sigmas, extra_args=None, callback=None,
                       disable=None, **kwargs):
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        _rms = lambda t: float(t.float().pow(2).mean().sqrt())

        if schedule_mode == "native" or (schedule_mode == "auto" and _native_av_schedule(model)):
            if verbose:
                print(f"[BSAI TaoMate Euler] native ModelSamplingAV -> 单调度 Euler  "
                      f"sigmas={[round(float(s), 4) for s in sigmas]}  x={tuple(x.shape)}",
                      flush=True)
            for i in trange(len(sigmas) - 1, disable=disable):
                sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
                denoised = model(x, sigmas[i] * s_in, **extra_args)
                d = (x - denoised) / sigmas[i]
                x = x + (sv_n - sv) * d
                if verbose:
                    print(f"[BSAI TaoMate step {i}] {sv:.4f}->{sv_n:.4f}  "
                          f"denoised_rms={_rms(denoised):.4f} x_rms={_rms(x):.4f}", flush=True)
                if callback is not None:
                    callback({"i": i, "denoised": denoised, "x": x,
                              "sigma": sigmas[i], "sigma_hat": sigmas[i]})
            return x

        # 旧版 ComfyUI：视频/音频各自 flow 调度（video shift 12 / audio shift 3）
        shapes = _latent_shapes(model)
        if not shapes or len(shapes) < 2:
            raise RuntimeError(
                "[BSAI TaoMate Euler] 需要 MiniMax-H3 视频+音频 latent "
                "(EmptyMiniMaxH3LatentAV / MiniMaxH3ImageToVideo 输出)。")
        v_numel = math.prod(shapes[0][1:])
        a_numel = x.shape[-1] - v_numel
        if verbose:
            print(f"[BSAI TaoMate Euler] legacy 双调度(无 ModelSamplingAV)  "
                  f"v_numel={v_numel} a_numel={a_numel} shapes={shapes}", flush=True)
        for i in trange(len(sigmas) - 1, disable=disable):
            sv, sv_n = float(sigmas[i]), float(sigmas[i + 1])
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            out = (x - denoised) / sigmas[i]
            xv, ov = x[..., :v_numel], out[..., :v_numel]
            xa, oa = x[..., v_numel:], out[..., v_numel:]
            xv = xv + (sv_n - sv) * ov
            sl = _audio_slope(max(sv, 1e-6), shift_video, shift_audio)
            xa = xa + (_audio_sigma(sv_n, shift_video, shift_audio)
                       - _audio_sigma(sv, shift_video, shift_audio)) * (oa / sl)
            x = torch.cat([xv, xa], dim=-1)
            if verbose:
                print(f"[BSAI TaoMate step {i}] {sv:.4f}->{sv_n:.4f}  "
                      f"video_rms={_rms(xv):.4f} audio_rms={_rms(xa):.4f} slope={sl:.4f}",
                      flush=True)
            if callback is not None:
                callback({"i": i, "denoised": denoised, "x": x,
                          "sigma": sigmas[i], "sigma_hat": sigmas[i]})
        return x
    return _taomate_euler


def _audio_sample_rate(audio_vae):
    return getattr(audio_vae, "audio_sample_rate_output",
                   getattr(audio_vae, "audio_sample_rate", 44100))


def _decode_audio(audio_vae, audio_latent):
    """按原生 VAEDecodeAudio 同款逻辑解码音频 latent -> [1,C,L]。"""
    audio = audio_vae.decode(audio_latent).movedim(-1, 1)
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = audio / std
    return audio


def _build_chunk_conditioning(clip, vae, audio_vae, prompt, width, height, length,
                              first_frame_img=None, tail_audio=None):
    """构造单个分块的 conditioning + AV latent（等价原生 ImageToVideo + AddGuide 音频锚）。

    - first_frame_img: 上一块末帧 -> 本块 frame 0 的图像锚（保持外观/场景）
    - tail_audio:      上一块尾音频 -> 本块 frame 0 的音频锚（保持声线/声音场景）
    """
    latent, frame_count = _empty_av_latent(width, height, length)

    images, keyframes = [], []
    if first_frame_img is not None:
        img = _resize(first_frame_img[:1], width, height, "disabled")
        images.append(img)
        keyframes.append({"resolved_frame_index": 0, "image": img})

    tokens = clip.tokenize(prompt, images=images)
    cond = clip.encode_from_tokens_scheduled(tokens)

    if keyframes:
        for kf in keyframes:
            kf["latent"] = vae.encode(kf.pop("image"))
        cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes})

    if tail_audio is not None and audio_vae is not None:
        kf_list = list(cond[0][1].get("minimax_keyframes", []))
        waveform = tail_audio["waveform"]
        sr = tail_audio["sample_rate"]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if sr != vae_sr:
            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
        audio_latent = audio_vae.encode(waveform[:1].movedim(1, -1))   # [1,32,2,T]
        kf_list.append({"resolved_frame_index": 0, "audio_latent": audio_latent})
        cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": kf_list})

    return cond, latent, frame_count


# ---------------------------------------------------------------------------
# 1) BSAITaoMateLoRALoader
# ---------------------------------------------------------------------------

class BSAITaoMateLoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        taomate = sorted(n for n in loras if "taomate" in os.path.splitext(n)[0].lower())
        combo = taomate if taomate else sorted(loras)
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (combo, {
                "tooltip": "TaoMate-H3 3 步蒸馏 LoRA（自动列出 loras 目录中的 TaoMate 文件）",
            }),
            "strength": ("FLOAT", {
                "default": 1.0, "min": -2.0, "max": 4.0, "step": 0.01,
                "tooltip": "LoRA 强度，TaoMate 官方默认 1.0",
            }),
            "strict_taomate_check": ("BOOLEAN", {
                "default": True, "label_on": "校验", "label_off": "关闭",
                "tooltip": "开启时仅接受文件名含 TaoMate 的 LoRA，防止误挂其它 LoRA",
            }),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "apply_lora"
    CATEGORY = "BSAI/TaoMate"
    DESCRIPTION = ("加载 TaoMate-H3 3 步蒸馏 LoRA 并应用到 MiniMax-H3 FL2VA 模型。"
                   "键名与原生 H3 模型完全匹配（含 token_refiner），走 ComfyUI 标准加载路径。")

    def apply_lora(self, model, lora_name, strength=1.0, strict_taomate_check=True):
        base = os.path.splitext(os.path.basename(lora_name))[0]
        if strict_taomate_check and "taomate" not in base.lower():
            raise ValueError(
                f"[BSAI TaoMate] {lora_name} 不是 TaoMate LoRA（文件名需含 TaoMate）；"
                "如确需强制加载请关闭 strict_taomate_check")

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        md = _read_safetensors_metadata(path)

        # 键匹配统计（诊断信息）
        key_map = comfy.lora.model_lora_keys_unet(model.model)
        target_keys = set()
        for k in key_map.keys():
            target_keys.add(k)
        lora = comfy.utils.load_torch_file(path)
        matched = sum(1 for k in lora.keys()
                      if k.endswith((".lora_down.weight", ".lora_up.weight"))
                      and k[: -len(".lora_down.weight" if k.endswith(".lora_down.weight") else ".lora_up.weight")] in target_keys)
        missing = [k for k in lora.keys()
                   if k.endswith((".lora_down.weight", ".lora_up.weight"))
                   and k[: -len(".lora_down.weight" if k.endswith(".lora_down.weight") else ".lora_up.weight")] not in target_keys]

        print(f"[BSAI TaoMate LoRALoader] {lora_name} strength={strength} "
              f"matched_keys={matched} unmatched={len(missing)} "
              f"base_model={(md or {}).get('base_model', '?')} "
              f"steps={(md or {}).get('optimizer_step', '?')}", flush=True)
        if missing:
            print(f"[BSAI TaoMate LoRALoader] 警告：{len(missing)} 个键无法匹配底模"
                  f"（前 8 个：{missing[:8]}）", flush=True)

        model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 1.0)
        return (model,)


# ---------------------------------------------------------------------------
# 2) BSAITaoMateTimesteps
# ---------------------------------------------------------------------------

class BSAITaoMateTimesteps:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "ladder": ("STRING", {
                "default": "999,750,500", "multiline": False,
                "tooltip": "TaoMate 3 步显式训练阶梯 timestep（0-1000），逗号分隔，"
                           "末尾自动补 0（最终去噪）。3 步默认 999,750,500",
            }),
        }}

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "BSAI/TaoMate"
    DESCRIPTION = ("把 TaoMate 显式训练阶梯 timestep 换算成 SamplerCustomAdvanced "
                   "的 SIGMAS。默认 [999,750,500]，3 步流式生成。")

    def get_sigmas(self, model, ladder="999,750,500"):
        values = []
        for part in str(ladder).replace("，", ",").replace("[", "").replace("]", "").split(","):
            part = part.strip()
            if part:
                values.append(float(part))
        if len(values) < 2:
            raise ValueError(f"[BSAI TaoMate Timesteps] 阶梯至少需要 2 个 timestep: {ladder!r}")
        ms = model.get_model_object("model_sampling")
        sig = [float(ms.sigma(torch.tensor(t, dtype=torch.float32))) for t in values]
        sig.append(0.0)                                    # 最后一步 -> 干净 latent
        sigmas = torch.tensor(sig, dtype=torch.float32, device="cpu")
        print(f"[BSAI TaoMate Timesteps] ladder={values} -> sigmas="
              f"{[round(float(s), 4) for s in sigmas]}", flush=True)
        return (sigmas,)


# ---------------------------------------------------------------------------
# 3) BSAITaoMateEulerSampler
# ---------------------------------------------------------------------------

class BSAITaoMateEulerSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shift_video": ("FLOAT", {
                "default": SHIFT_V, "min": 0.01, "max": 100.0, "step": 0.01,
                "tooltip": "视频流 flow shift，H3 官方 12.0",
            }),
            "shift_audio": ("FLOAT", {
                "default": SHIFT_A, "min": 0.01, "max": 100.0, "step": 0.01,
                "tooltip": "音频流 flow shift，H3 官方 3.0",
            }),
            "schedule_mode": (["auto", "native", "legacy_dual"], {
                "default": "auto",
                "tooltip": "auto: 新版 ComfyUI（ModelSamplingAV）单调度 Euler；"
                           "旧版无 ModelSamplingAV 时自动退化为音视频双调度",
            }),
            "verbose": ("BOOLEAN", {"default": True, "label_on": "打印", "label_off": "静默"}),
        }}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "BSAI/TaoMate"
    DESCRIPTION = ("TaoMate 3 步 Euler 采样器（音视频双调度自适应），"
                   "供 SamplerCustomAdvanced 使用。")

    def get_sampler(self, shift_video=SHIFT_V, shift_audio=SHIFT_A,
                    schedule_mode="auto", verbose=True):
        sampler = comfy.samplers.KSAMPLER(
            make_taomate_euler(shift_video, shift_audio, schedule_mode, verbose))
        return (sampler,)


# ---------------------------------------------------------------------------
# 4) BSAITaoMateContinuationProbe
# ---------------------------------------------------------------------------

class BSAITaoMateContinuationProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "vae": ("VAE",),
            "audio_vae": ("VAE",),
            "tail_audio_seconds": ("FLOAT", {
                "default": 1.5, "min": 0.2, "max": 6.0, "step": 0.1,
                "tooltip": "提取尾音频长度（秒），作为下一块的音频锚",
            }),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("last_frame", "tail_audio", "frame_count")
    FUNCTION = "probe"
    CATEGORY = "BSAI/TaoMate"
    DESCRIPTION = ("从已完成分块的 H3 AV latent 中提取末帧 + 尾音频，"
                   "供下一块 MiniMaxH3ImageToVideo(first_frame) / "
                   "MiniMaxH3AddGuide(audio) 续接。")

    def probe(self, latent, vae, audio_vae, tail_audio_seconds=1.5):
        samples = latent["samples"]
        if not samples.is_nested or len(samples.tensors) != 2:
            raise ValueError("[BSAI TaoMate ContinuationProbe] 需要 MiniMax-H3 视频+音频 "
                             "latent（EmptyMiniMaxH3LatentAV / MiniMaxH3ImageToVideo 输出）。")
        video_part, audio_part = samples.unbind()[0], samples.unbind()[-1]

        frames = vae.decode(video_part).movedim(1, -1)          # [1,T,H,W,C]
        last_frame = frames[0, -1:].clone()                     # [1,H,W,C]

        audio = _decode_audio(audio_vae, audio_part)            # [1,C,L]
        sr = _audio_sample_rate(audio_vae)
        L = int(tail_audio_seconds * sr)
        tail = audio[..., -L:].clone()

        frame_count = int(frames.shape[1])
        print(f"[BSAI TaoMate ContinuationProbe] frames={frame_count} "
              f"tail_audio={tail.shape[-1] / sr:.2f}s @ {sr}Hz", flush=True)
        return (last_frame, {"waveform": tail, "sample_rate": sr}, frame_count)


# ---------------------------------------------------------------------------
# 5) BSAITaoMateStreamChain
# ---------------------------------------------------------------------------

class BSAITaoMateStreamChain:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "audio_vae": ("VAE",),
            "sigmas": ("SIGMAS",),
            "prompts": ("STRING", {
                "multiline": True,
                "default": ("场景 1：……（每行一段提示词，一段=一个分块）\n"
                             "场景 2：……\n场景 3：……"),
                "tooltip": "每行一个分块提示词；每个分块生成一段音视频并自动以"
                           "上一块末帧+尾音频续接，实现外观/声线连续的长视频",
            }),
            "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
            "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
            "length": ("INT", {
                "default": 124, "min": 5, "max": 362, "step": 17,
                "tooltip": "单块帧数（24fps，自动吸附 17k+5 网格；124≈5.2s，"
                           "训练区间约 124-362）",
            }),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "anchor_tail_audio": ("BOOLEAN", {
                "default": True, "label_on": "锚定", "label_off": "不锚定",
                "tooltip": "把上一块尾音频作为下一块 frame 0 的音频锚，保持声线/声音场景",
            }),
            "tail_audio_seconds": ("FLOAT", {
                "default": 1.5, "min": 0.2, "max": 6.0, "step": 0.1,
            }),
            "shift_video": ("FLOAT", {"default": SHIFT_V, "min": 0.01, "max": 100.0, "step": 0.01}),
            "shift_audio": ("FLOAT", {"default": SHIFT_A, "min": 0.01, "max": 100.0, "step": 0.01}),
            "verbose": ("BOOLEAN", {"default": True, "label_on": "打印", "label_off": "静默"}),
        }, "optional": {
            "first_frame": ("IMAGE", {"tooltip": "第一块的起始锚图（可选）"}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("video", "audio", "last_frame", "tail_audio", "chunks")
    FUNCTION = "stream"
    CATEGORY = "BSAI/TaoMate"
    DESCRIPTION = ("TaoMate 一键流式长视频：按 prompts 逐块生成，每块 3 步采样，"
                   "自动以上一块末帧+尾音频续接下一块，输出拼接后的完整音视频。"
                   "cfg 固定 1.0（3 步蒸馏模型不做 CFG）。")

    def stream(self, model, clip, vae, audio_vae, sigmas, prompts,
               width=1344, height=768, length=124, seed=0,
               anchor_tail_audio=True, tail_audio_seconds=1.5,
               shift_video=SHIFT_V, shift_audio=SHIFT_A, verbose=True,
               first_frame=None):
        lines = [p.strip() for p in prompts.splitlines() if p.strip()]
        if not lines:
            raise ValueError("[BSAI TaoMate StreamChain] prompts 为空：每行一个分块提示词")

        sampler = comfy.samplers.KSAMPLER(
            make_taomate_euler(shift_video, shift_audio, "auto", verbose))
        sr = _audio_sample_rate(audio_vae)

        all_frames, all_audio = [], []
        cur_first_frame = first_frame
        cur_tail_audio = None

        for i, prompt in enumerate(lines):
            cond, latent, frame_count = _build_chunk_conditioning(
                clip, vae, audio_vae, prompt, width, height, length,
                cur_first_frame, cur_tail_audio if anchor_tail_audio else None)

            noise = Noise_RandomNoise(seed + i)
            guider = Guider_Basic(model)
            guider.set_conds(cond)

            latent_image = latent["samples"]
            latent_image = comfy.sample.fix_empty_latent_channels(
                guider.model_patcher, latent_image,
                latent.get("downscale_ratio_spacial", None),
                latent.get("downscale_ratio_temporal", None))

            # BSAI 协同：GPU1 采样期间持有租约（allocate 失败不改变原流程）
            _s_alloc = None
            try:
                if BSAIOrch is not None:
                    _s_alloc = BSAIOrch.allocate("sampling", requester="8191", watchdog=True)
            except Exception:
                _s_alloc = None
            try:
                samples = guider.sample(
                    noise.generate_noise(latent), latent_image, sampler, sigmas,
                    denoise_mask=None, callback=None,
                    disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                    seed=noise.seed)
            finally:
                if _s_alloc is not None:
                    try:
                        _s_alloc.release()
                    except Exception:
                        pass
            samples = samples.to(comfy.model_management.intermediate_device())

            video_part, audio_part = samples.unbind()[0], samples.unbind()[-1]
            frames = vae.decode(video_part).movedim(1, -1)      # [1,T,H,W,C]
            audio = _decode_audio(audio_vae, audio_part)        # [1,C,L]

            all_frames.append(frames[0])
            all_audio.append(audio[0])

            cur_first_frame = frames[0, -1:]                    # 末帧 -> 下一块首帧锚
            L = int(tail_audio_seconds * sr)
            cur_tail_audio = {"waveform": audio[..., -L:], "sample_rate": sr}

            if verbose:
                print(f"[BSAI TaoMate StreamChain] 块 {i + 1}/{len(lines)} "
                      f"frames={frames.shape[1]} audio={audio.shape[-1] / sr:.2f}s @ {sr}Hz",
                      flush=True)

        video = torch.cat(all_frames, dim=0)                     # [T_total,H,W,C]
        waveform = torch.cat(all_audio, dim=-1).unsqueeze(0)     # [1,C,L_total]
        tail = {"waveform": cur_tail_audio["waveform"], "sample_rate": sr}
        print(f"[BSAI TaoMate StreamChain] 完成 {len(lines)} 块 -> "
              f"video_frames={video.shape[0]} audio={waveform.shape[-1] / sr:.2f}s",
              flush=True)
        return (video, {"waveform": waveform, "sample_rate": sr},
                cur_first_frame, tail, len(lines))


NODE_CLASS_MAPPINGS = {
    "BSAITaoMateLoRALoader": BSAITaoMateLoRALoader,
    "BSAITaoMateTimesteps": BSAITaoMateTimesteps,
    "BSAITaoMateEulerSampler": BSAITaoMateEulerSampler,
    "BSAITaoMateContinuationProbe": BSAITaoMateContinuationProbe,
    "BSAITaoMateStreamChain": BSAITaoMateStreamChain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAITaoMateLoRALoader": "BSAI TaoMate LoRA Loader (3-Step)",
    "BSAITaoMateTimesteps": "BSAI TaoMate 3-Step Timesteps",
    "BSAITaoMateEulerSampler": "BSAI TaoMate Euler Sampler (3-Step)",
    "BSAITaoMateContinuationProbe": "BSAI TaoMate Continuation Probe",
    "BSAITaoMateStreamChain": "BSAI TaoMate Stream Chain (Long Video)",
}
