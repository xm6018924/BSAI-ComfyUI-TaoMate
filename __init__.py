"""
BSAI-ComfyUI-TaoMate
====================

ComfyUI 节点包：把阿里淘宝直播团队开源的 TaoMate-H3（基于 MiniMax H3 的
低延迟流式音视频生成，3 步蒸馏 LoRA + KV 缓存流式续写）封装为原生
ComfyUI 节点与可运行工作流。

视频中 TaoMate-H3 目前"不是 ComfyUI 自定义节点、也没有 ComfyUI 支持"，
本插件即补上这一环：

  * BSAITaoMateLoRALoader        3 步蒸馏 LoRA 加载（TaoMate-H3-step3000）
  * BSAITaoMateTimesteps         3 步显式时间阶梯 -> SIGMAS
  * BSAITaoMateEulerSampler      音视频双调度 Euler 采样器（3 步）
  * BSAITaoMateContinuationProbe 块续接探针：取上一块末帧 + 尾音频
  * BSAITaoMateStreamChain       一键流式长视频：多块连续生成并自动续接

所有节点仅通过 ModelPatcher.clone() / model_options 注入，不改 ComfyUI
内部源码，升级无碍。
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
