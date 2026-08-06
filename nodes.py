from __future__ import annotations

import json
import math
import re
from typing import Any

import torch

from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_minimax_h3 import (
    FPS,
    MiniMaxH3ImageToVideo,
    MiniMaxH3ReferenceToVideo,
    temporal_shape,
)

from .media import (
    load_audio,
    load_image,
    load_video,
    normalize_audio_for_h3,
    validate_generation_size,
    validate_image_tensor,
    validate_reference_duration,
    validate_reference_limits,
)

MIN_OUTPUT_SECONDS = 0.0
DEFAULT_DURATION_SECONDS = 124 / FPS
SLOT_LIMITS = {
    "ref_image": 9,
    "ref_video": 3,
    "ref_video_audio": 3,
    "ref_audio": 3,
}


def parse_media_state(value: str | dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if value in (None, ""):
        return {}
    try:
        state = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("MiniMax H3 媒体状态不是有效 JSON") from error
    if not isinstance(state, dict):
        raise ValueError("MiniMax H3 媒体状态必须是 JSON 对象")
    return {str(key): item for key, item in state.items() if isinstance(item, dict)}


def resolve_slots(
    prefix: str,
    count: int,
    state: dict[str, dict[str, Any]],
    external: dict[str, Any],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for index in range(1, count + 1):
        slot = f"{prefix}_{index}"
        external_value = external.get(slot)
        if external_value is not None:
            resolved[slot] = external_value
        elif state.get(slot, {}).get("path"):
            resolved[slot] = state[slot]
    return resolved


def autogrow_slots(prefix: str, values: io.Autogrow.Type | None) -> dict[str, Any]:
    """Map zero-based Autogrow child names to one-based internal slot names.

    Sparse inputs retain their original index so a missing middle socket cannot
    shift a later image/video/audio into the wrong pairing.
    """
    resolved: dict[str, Any] = {}
    unnamed: list[Any] = []
    pattern = re.compile(rf"(?:^|\.){re.escape(prefix)}_(\d+)$")
    for name, value in (values or {}).items():
        if value is None:
            continue
        match = pattern.search(str(name))
        if match:
            resolved[f"{prefix}_{int(match.group(1)) + 1}"] = value
        else:
            unnamed.append(value)

    next_index = 1
    for value in unnamed:
        while f"{prefix}_{next_index}" in resolved:
            next_index += 1
        resolved[f"{prefix}_{next_index}"] = value
        next_index += 1
    return resolved


def _prepare_external_audio(value: dict[str, Any], slot: str) -> tuple[dict[str, Any], float]:
    waveform = value.get("waveform")
    sample_rate = int(value.get("sample_rate", 0))
    if not torch.is_tensor(waveform) or waveform.ndim != 3 or sample_rate <= 0 or waveform.shape[-1] <= 0:
        raise ValueError(f"{slot} 必须是非空 ComfyUI AUDIO")
    duration = int(waveform.shape[-1]) / sample_rate
    validate_reference_duration(duration, f"{slot}")
    return normalize_audio_for_h3(value), duration


def _prepare_audio_slots(
    slots: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    prepared: dict[str, dict[str, Any]] = {}
    durations: dict[str, float] = {}
    for slot, value in slots.items():
        if isinstance(value, dict) and "waveform" in value:
            audio, duration = _prepare_external_audio(value, slot)
        else:
            audio, duration = load_audio(value)
        prepared[slot] = audio
        durations[slot] = float(duration)
    return prepared, durations


def _state_trim_duration(item: dict[str, Any], label: str) -> float:
    total = 0.0
    for key in ("video_duration", "duration"):
        try:
            candidate = float(item.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            candidate = 0.0
        if math.isfinite(candidate) and candidate > 0:
            total = candidate
            break
    if total <= 0:
        raise ValueError(f"{label}缺少可用时长元数据，请重新上传该媒体")

    try:
        start = float(item.get("trim_start", 0.0) or 0.0)
        raw_end = float(item.get("trim_end", 0.0) or 0.0)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}裁剪时间不是有效数字") from error
    if not math.isfinite(start) or not math.isfinite(raw_end):
        raise ValueError(f"{label}裁剪时间必须是有限数字")
    start = min(total, max(0.0, start))
    end = total if raw_end <= 0 else min(total, max(0.0, raw_end))
    duration = end - start
    if duration <= 0:
        raise ValueError(f"{label}裁剪区间无效：{start:.3f}s–{end:.3f}s")
    validate_reference_duration(duration, label)
    return duration


def _embedded_audio_duration_candidates(
    video_slots: dict[str, Any],
    explicit_paired_audio: dict[str, Any],
) -> list[float]:
    durations: list[float] = []
    for slot, value in video_slots.items():
        paired_slot = slot.replace("ref_video_", "ref_video_audio_")
        if paired_slot in explicit_paired_audio:
            continue
        if (
            isinstance(value, dict)
            and value.get("use_audio")
            and value.get("has_audio") is not False
        ):
            # New uploads persist has_audio=False for silent videos. Older
            # workflows may not have this key, so None remains probe-compatible.
            durations.append(_state_trim_duration(value, f"{slot} 原声"))
    return durations


def _normalize_manual_duration(value: float | int) -> float:
    """Validate the public duration input, which is always expressed in seconds."""
    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("duration 必须是有效秒数") from error
    if not math.isfinite(seconds):
        raise ValueError("duration 必须是有限数字")
    if seconds < 0:
        raise ValueError(f"duration 不能小于 0 秒；当前为 {seconds:.3f} 秒")
    return seconds


def _seconds_to_requested_frames(seconds: float) -> int:
    raw_frames = seconds * FPS
    if not math.isfinite(raw_frames):
        raise ValueError("duration 数值过大，无法换算为帧数")
    return max(0, int(round(raw_frames)))


def _resolve_frame_count(
    duration_seconds: float,
    auto_length_from_audio: bool,
    audio_durations: list[float],
) -> int:
    if auto_length_from_audio:
        if not audio_durations:
            raise ValueError("已启用按音频自动长度，但当前没有可用参考音频")
        seconds = max(float(value) for value in audio_durations)
    else:
        seconds = _normalize_manual_duration(duration_seconds)

    requested_length = _seconds_to_requested_frames(seconds)
    frame_count, _video_latent_t, _audio_latent_t = temporal_shape(requested_length)
    return int(frame_count)


def _select_reference_audio(
    ordered_audio: list[tuple[str, dict[str, Any], float]],
) -> dict[str, Any] | None:
    """Return one standard AUDIO object for downstream preview/save nodes.

    Ref2VA accepts several audio references, but ComfyUI's AUDIO socket carries a
    single object. The longest effective reference is the least surprising
    deterministic choice and is also the source used by automatic duration.
    """
    if not ordered_audio:
        return None
    return max(ordered_audio, key=lambda item: item[2])[1]


class MiniMaxH3Unified(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Unified",
            display_name="MiniMax H3 Unified",
            category="MiniMax/H3",
            description="统一路由 MiniMax H3 FL2VA 与 Ref2VA，并支持节点内媒体上传。",
            inputs=[
                io.DynamicCombo.Input(
                    "mode",
                    tooltip="首尾帧与参考媒体接口按当前模式动态显示。clip、vae、audio_vae 始终显示。",
                    options=[
                        io.DynamicCombo.Option("text_to_video", []),
                        io.DynamicCombo.Option(
                            "first_last_frame",
                            [
                                io.Image.Input("first_frame", optional=True),
                                io.Image.Input("last_frame", optional=True),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            "omni_reference",
                            [
                                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                                io.Autogrow.Input(
                                    "ref_images",
                                    optional=True,
                                    template=io.Autogrow.TemplatePrefix(
                                        input=io.Image.Input("ref_image"),
                                        prefix="ref_image_",
                                        min=0,
                                        max=9,
                                    ),
                                ),
                                io.Autogrow.Input(
                                    "ref_videos",
                                    optional=True,
                                    template=io.Autogrow.TemplatePrefix(
                                        input=io.Image.Input("ref_video", tooltip="24 FPS IMAGE 批次"),
                                        prefix="ref_video_",
                                        min=0,
                                        max=3,
                                    ),
                                ),
                                io.Autogrow.Input(
                                    "ref_video_audios",
                                    optional=True,
                                    tooltip="同编号参考视频的配对音轨组",
                                    template=io.Autogrow.TemplatePrefix(
                                        input=io.Audio.Input("ref_video_audio", tooltip="同编号参考视频的替换音轨"),
                                        prefix="ref_video_audio_",
                                        min=0,
                                        max=3,
                                    ),
                                ),
                                io.Autogrow.Input(
                                    "ref_audios",
                                    optional=True,
                                    tooltip="独立参考音频组",
                                    template=io.Autogrow.TemplatePrefix(
                                        input=io.Audio.Input("ref_audio", tooltip="独立参考音频"),
                                        prefix="ref_audio_",
                                        min=0,
                                        max=3,
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae", optional=True),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Float.Input(
                    "duration",
                    display_name="duration",
                    default=round(DEFAULT_DURATION_SECONDS, 3),
                    min=MIN_OUTPUT_SECONDS,
                    step=0.1,
                    round=0.001,
                    tooltip="输入目标秒数（允许 0，且不设上限）。节点会按 24 FPS 换算并向上对齐到 H3 的 17k+5 网格；0 秒会得到模型可接受的最小 5 帧。超长时长可能显著增加内存、显存和生成时间。",
                ),
                io.Boolean.Input(
                    "auto_length_from_audio",
                    display_name="有音频时自动长度",
                    default=False,
                    label_on="启用",
                    label_off="关闭",
                    tooltip="仅在 omni_reference 且存在有效参考音频时，按最长音频自动计算时长；开启后手动时长控件会变灰且不可操作。其他模式或无音频时回退到手动秒数。",
                ),
                io.String.Input("media_state", default="{}"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="av_latent"),
                io.Audio.Output(
                    display_name="audio",
                    tooltip=(
                        "输出当前实际参与参考的最长一段裁剪后音频；无参考音频时为 None。"
                        "这是参考音频，不是采样后生成的视频音轨。"
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        mode: str | dict[str, Any],
        clip: Any,
        vae: Any,
        audio_vae: Any = None,
        prompt: str = "",
        width: int = 1344,
        height: int = 768,
        duration: float = DEFAULT_DURATION_SECONDS,
        auto_length_from_audio: bool = False,
        ref_image_size: str = "match",
        media_state: str = "{}",
        first_frame: torch.Tensor | None = None,
        last_frame: torch.Tensor | None = None,
        ref_images: io.Autogrow.Type | None = None,
        ref_videos: io.Autogrow.Type | None = None,
        ref_video_audios: io.Autogrow.Type | None = None,
        ref_audios: io.Autogrow.Type | None = None,
    ):
        if isinstance(mode, dict):
            values = mode
            mode = str(values.get("mode", ""))
            audio_vae = values.get("audio_vae", audio_vae)
            first_frame = values.get("first_frame", first_frame)
            last_frame = values.get("last_frame", last_frame)
            ref_image_size = values.get("ref_image_size", ref_image_size)
            auto_length_from_audio = bool(values.get("auto_length_from_audio", auto_length_from_audio))
            ref_images = values.get("ref_images", ref_images)
            ref_videos = values.get("ref_videos", ref_videos)
            ref_video_audios = values.get("ref_video_audios", ref_video_audios)
            ref_audios = values.get("ref_audios", ref_audios)

        external: dict[str, Any] = {}
        for prefix, values in (
            ("ref_image", ref_images),
            ("ref_video", ref_videos),
            ("ref_video_audio", ref_video_audios),
            ("ref_audio", ref_audios),
        ):
            external.update(autogrow_slots(prefix, values))

        width = int(width)
        height = int(height)
        validate_generation_size(width, height)
        state = parse_media_state(media_state)
        if mode in {"text_to_video", "first_last_frame"}:
            frame_count = _resolve_frame_count(duration, False, [])
            if mode == "first_last_frame":
                if first_frame is None and state.get("first_frame", {}).get("path"):
                    first_frame = load_image(state["first_frame"])
                if last_frame is None and state.get("last_frame", {}).get("path"):
                    last_frame = load_image(state["last_frame"])
                if first_frame is None and last_frame is None:
                    raise ValueError("first_last_frame 模式至少需要首帧或尾帧")
            else:
                first_frame = None
                last_frame = None
            if first_frame is not None:
                first_frame = validate_image_tensor(first_frame, "首帧")
            if last_frame is not None:
                last_frame = validate_image_tensor(last_frame, "尾帧")
            positive, latent = MiniMaxH3ImageToVideo.execute(
                clip,
                vae,
                prompt,
                width,
                height,
                frame_count,
                first_frame,
                last_frame,
            )[:2]
            return io.NodeOutput(positive, latent, None)


        if mode != "omni_reference":
            raise ValueError(f"未知模式：{mode}")
        if ref_image_size not in {"match", "max"}:
            raise ValueError(f"未知参考图尺寸模式：{ref_image_size}")

        image_slots = resolve_slots("ref_image", SLOT_LIMITS["ref_image"], state, external)
        video_slots = resolve_slots("ref_video", SLOT_LIMITS["ref_video"], state, external)
        paired_slots = resolve_slots("ref_video_audio", SLOT_LIMITS["ref_video_audio"], state, external)
        audio_slots = resolve_slots("ref_audio", SLOT_LIMITS["ref_audio"], state, external)

        # Reject orphan paired-audio sockets before decoding them. Besides a
        # clearer error, this prevents wasting CPU/RAM on audio that cannot be
        # consumed by the official same-index video/audio pairing.
        expected_paired_slots = {
            slot.replace("ref_video_", "ref_video_audio_") for slot in video_slots
        }
        orphan_paired_slots = set(paired_slots) - expected_paired_slots
        if orphan_paired_slots:
            raise ValueError(
                "配对音频缺少同编号视频：" + ", ".join(sorted(orphan_paired_slots))
            )

        # Decode explicit audio once. Its exact trimmed duration is then available
        # before AV latent creation, so the node can calculate H3 length internally.
        paired_audio, paired_duration_by_slot = _prepare_audio_slots(paired_slots)
        audios, standalone_duration_by_slot = _prepare_audio_slots(audio_slots)
        length_candidates = list(paired_duration_by_slot.values()) + list(standalone_duration_by_slot.values())
        auto_length_requested = bool(auto_length_from_audio)
        if auto_length_requested:
            length_candidates.extend(_embedded_audio_duration_candidates(video_slots, paired_audio))
        # The audio-derived policy is intentionally mode-aware: text-to-video,
        # first/last-frame, and reference workflows without usable audio keep the
        # manual length. This avoids forcing users to add dummy audio or toggle
        # the option off when switching among the unified node's modes.
        use_audio_length = auto_length_requested and bool(length_candidates)
        frame_count = _resolve_frame_count(
            duration,
            use_audio_length,
            length_candidates,
        )

        images = {
            slot: validate_image_tensor(value, slot) if torch.is_tensor(value) else load_image(value)
            for slot, value in image_slots.items()
        }

        videos: dict[str, torch.Tensor] = {}
        video_durations: list[float] = []
        embedded_audio_count = 0
        for slot, value in video_slots.items():
            if torch.is_tensor(value):
                frames = validate_image_tensor(value, slot)
                selected_duration = int(frames.shape[0]) / FPS
                soundtrack = None
                validate_reference_duration(selected_duration, slot)
            else:
                frames, soundtrack, selected_duration, _selected_frames = load_video(value, frame_count)
            videos[slot] = frames
            video_durations.append(selected_duration)
            paired_slot = slot.replace("ref_video_", "ref_video_audio_")
            if paired_slot not in paired_audio and soundtrack is not None:
                soundtrack_duration = int(soundtrack["waveform"].shape[-1]) / int(soundtrack["sample_rate"])
                validate_reference_duration(soundtrack_duration, paired_slot)
                paired_audio[paired_slot] = soundtrack
                paired_duration_by_slot[paired_slot] = soundtrack_duration
                embedded_audio_count += 1

        audio_durations = list(paired_duration_by_slot.values()) + list(standalone_duration_by_slot.values())
        ordered_reference_audio: list[tuple[str, dict[str, Any], float]] = []
        for video_slot in videos:
            paired_slot = video_slot.replace("ref_video_", "ref_video_audio_")
            if paired_slot in paired_audio:
                ordered_reference_audio.append(
                    (paired_slot, paired_audio[paired_slot], paired_duration_by_slot[paired_slot])
                )
        for audio_slot in audios:
            ordered_reference_audio.append(
                (audio_slot, audios[audio_slot], standalone_duration_by_slot[audio_slot])
            )

        validate_reference_limits(
            images,
            videos,
            paired_audio,
            audios,
            video_durations,
            audio_durations,
            embedded_audio_count,
        )
        if ordered_reference_audio and audio_vae is None:
            raise ValueError("包含参考音频时必须连接 audio_vae")

        output = MiniMaxH3ReferenceToVideo.execute(
            clip,
            vae,
            audio_vae,
            prompt,
            width,
            height,
            frame_count,
            ref_image_size,
            images,
            videos,
            paired_audio,
            audios,
        )
        positive, latent = output[0], output[1]
        return io.NodeOutput(
            positive,
            latent,
            _select_reference_audio(ordered_reference_audio),
        )



class MiniMaxH3UnifiedExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3Unified]


async def comfy_entrypoint():
    return MiniMaxH3UnifiedExtension()
