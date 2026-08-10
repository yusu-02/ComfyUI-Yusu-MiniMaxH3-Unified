from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import folder_paths
from comfy_extras.nodes_minimax_h3 import FPS, adapt_canvas

MEDIA_SUBDIR = "minimax_h3_unified"
CANVAS_MULTIPLE = 32
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000
MAX_AUDIO_SAMPLE_RATE = 384_000
MIN_REFERENCE_SECONDS = 2.0
MAX_REFERENCE_SECONDS = 15.0
GENERIC_MIME = {"", "application/octet-stream", "binary/octet-stream"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
ALLOWED_MIME = {
    ".png": {"image/png"},
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".webp": {"image/webp"},
    ".mp4": {"video/mp4", "application/mp4"},
    ".webm": {"video/webm", "audio/webm"},
    ".mov": {"video/quicktime", "video/mp4"},
    ".mkv": {"video/x-matroska", "application/x-matroska", "video/mkv"},
    ".wav": {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"},
    ".mp3": {"audio/mpeg"},
    ".flac": {"audio/flac", "audio/x-flac"},
    ".ogg": {"audio/ogg", "application/ogg"},
    ".m4a": {"audio/mp4", "audio/x-m4a"},
    ".aac": {"audio/aac", "audio/x-aac", "audio/aacp"},
}


def finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}必须是有限数字") from error
    if not np.isfinite(number):
        raise ValueError(f"{label}必须是有限数字")
    return number


def positive_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) and number > 0 else 0.0


def validate_generation_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("生成宽高必须为正整数")
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        suggested_width = max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        suggested_height = max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        raise ValueError(
            f"MiniMax H3 生成宽高必须都是 {CANVAS_MULTIPLE} 的倍数；"
            f"当前为 {width}×{height}，建议改为 {suggested_width}×{suggested_height}"
        )


def validate_reference_duration(duration: float, label: str = "参考媒体") -> None:
    if not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
        raise ValueError(f"{label}必须为 {MIN_REFERENCE_SECONDS:g}–{MAX_REFERENCE_SECONDS:g} 秒")


def validate_reference_audio_duration(duration: float, label: str = "参考音频") -> None:
    if not np.isfinite(duration) or not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
        raise ValueError(f"{label}必须为 {MIN_REFERENCE_SECONDS:g}–{MAX_REFERENCE_SECONDS:g} 秒")


def normalize_audio_for_h3(audio: dict[str, Any]) -> dict[str, Any]:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not torch.is_tensor(waveform) or waveform.ndim != 3 or sample_rate <= 0 or waveform.shape[-1] <= 0:
        raise ValueError("参考音频必须是非空 waveform [B,C,L] 和正采样率")
    if sample_rate > MAX_AUDIO_SAMPLE_RATE:
        raise ValueError(f"参考音频采样率过高：{sample_rate} Hz")
    channels = int(waveform.shape[1])
    if channels == 1:
        waveform = waveform[:1, :1, :].to(dtype=torch.float32).expand(-1, 2, -1).clone()
    elif channels >= 2:
        waveform = waveform[:1, :2, :].to(dtype=torch.float32)
    else:
        raise ValueError("参考音频不含有效声道")
    if not torch.isfinite(waveform).all().item():
        raise ValueError("参考音频包含 NaN 或无穷值")
    waveform = waveform.clamp(-1.0, 1.0).contiguous()
    if float(waveform.abs().amax().item()) <= 1e-7:
        raise ValueError("裁剪后的参考音频是静音，模型无法从中提取有效参考")
    return {"waveform": waveform, "sample_rate": sample_rate}


def media_root() -> Path:
    input_root = Path(folder_paths.get_input_directory()).resolve()
    root = (input_root / MEDIA_SUBDIR).resolve()
    if not root.is_relative_to(input_root):
        raise ValueError(f"媒体目录越界：{root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_media_path(relative_path: str) -> Path:
    candidate = Path(relative_path)
    annotated = relative_path.endswith(("[input]", "[output]", "[temp]"))
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts or annotated:
        raise ValueError(f"非法媒体路径：{relative_path}")
    try:
        path = Path(folder_paths.get_annotated_filepath(candidate.as_posix())).resolve()
    except ValueError as error:
        raise ValueError(f"媒体路径越界：{relative_path}") from error
    if not path.is_file():
        raise FileNotFoundError(f"媒体文件不存在：{relative_path}")
    return path


def validate_upload_name(filename: str, content_type: str) -> tuple[str, str]:
    if not filename or Path(filename).name != filename or any(character in filename for character in ("/", "\\", "\0")):
        raise ValueError("非法文件名")
    extension = Path(filename).suffix.lower()
    if extension not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS:
        raise ValueError(f"不支持的扩展名：{extension or '无'}")
    mime = content_type.split(";", 1)[0].lower().strip()
    if mime not in GENERIC_MIME and mime not in ALLOWED_MIME[extension]:
        raise ValueError(f"MIME {mime} 与扩展名 {extension} 不匹配")
    kind = "image" if extension in IMAGE_EXTENSIONS else "video" if extension in VIDEO_EXTENSIONS else "audio"
    return extension, kind


def _probe_with_pyav(path: Path) -> dict[str, Any]:
    with av.open(str(path)) as container:
        container_duration = (
            positive_float(container.duration / av.time_base)
            if container.duration is not None
            else 0.0
        )
        video_durations: list[float] = []
        audio_durations: list[float] = []
        for stream in container.streams:
            duration = (
                positive_float(stream.duration * stream.time_base)
                if stream.duration is not None and stream.time_base is not None
                else 0.0
            )
            if stream.type == "video":
                video_durations.append(duration)
            elif stream.type == "audio":
                audio_durations.append(duration)
        has_video = bool(container.streams.video)
        has_audio = bool(container.streams.audio)
        video_duration = max(video_durations, default=0.0) or (container_duration if has_video else 0.0)
        audio_duration = max(audio_durations, default=0.0) or (container_duration if has_audio else 0.0)
        return {
            "duration": max(container_duration, video_duration, audio_duration),
            "video_duration": video_duration,
            "audio_duration": audio_duration,
            "has_video": has_video,
            "has_audio": has_audio,
        }


def _duration_from_ffprobe_stream(stream: dict[str, Any]) -> float:
    candidates = [positive_float(stream.get("duration"))]
    duration_ts = positive_float(stream.get("duration_ts"))
    time_base = str(stream.get("time_base") or "")
    if duration_ts and "/" in time_base:
        try:
            numerator, denominator = (float(part) for part in time_base.split("/", 1))
        except ValueError:
            return max(candidates, default=0.0)
        if denominator:
            candidates.append(duration_ts * numerator / denominator)
    return max(candidates, default=0.0)


def probe_media(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    ffprobe_error: Exception | None = None
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration:stream=codec_type,duration,duration_ts,time_base",
                    "-of",
                    "json",
                    str(path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
                check=False,
            )
            if result.returncode:
                raise ValueError(f"ffprobe 无法读取媒体：{result.stderr.strip() or path.name}")
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if not isinstance(streams, list):
                raise ValueError(f"ffprobe 返回了无效流元数据：{path.name}")
            container_duration = positive_float(data.get("format", {}).get("duration"))
            video_durations = [
                _duration_from_ffprobe_stream(stream)
                for stream in streams
                if isinstance(stream, dict) and stream.get("codec_type") == "video"
            ]
            audio_durations = [
                _duration_from_ffprobe_stream(stream)
                for stream in streams
                if isinstance(stream, dict) and stream.get("codec_type") == "audio"
            ]
            metadata = {
                "duration": max(container_duration, *video_durations, *audio_durations, 0.0),
                "video_duration": max(video_durations, default=0.0),
                "audio_duration": max(audio_durations, default=0.0),
                "has_video": bool(video_durations),
                "has_audio": bool(audio_durations),
            }
            if (
                (metadata["has_video"] and metadata["video_duration"] <= 0)
                or (metadata["has_audio"] and metadata["audio_duration"] <= 0)
                or metadata["duration"] <= 0
            ):
                fallback = _probe_with_pyav(path)
                for key in ("duration", "video_duration", "audio_duration"):
                    metadata[key] = metadata[key] or fallback[key]
                metadata["has_video"] = metadata["has_video"] or fallback["has_video"]
                metadata["has_audio"] = metadata["has_audio"] or fallback["has_audio"]
            return metadata
        except (json.JSONDecodeError, OSError, subprocess.TimeoutExpired, ValueError) as error:
            ffprobe_error = error
    try:
        return _probe_with_pyav(path)
    except Exception as pyav_error:
        detail = f"ffprobe={ffprobe_error}; PyAV={pyav_error}" if ffprobe_error is not None else str(pyav_error)
        raise ValueError(f"无法读取媒体：{detail}") from pyav_error


def validate_uploaded_file(path: Path, kind: str) -> dict[str, Any]:
    if kind == "image":
        with Image.open(path) as image:
            expected = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}[path.suffix.lower()]
            if image.format != expected:
                raise ValueError(f"图片内容格式 {image.format or '未知'} 与扩展名不匹配")
            if image.width <= 0 or image.height <= 0 or image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"图片尺寸过大或无效：{image.width}×{image.height}")
            image.verify()
        return {"duration": 0.0, "has_video": False, "has_audio": False}
    metadata = probe_media(path)
    if kind == "video":
        if not metadata["has_video"]:
            raise ValueError("上传文件不含视频流")
        metadata["duration"] = metadata.get("video_duration") or metadata["duration"]
    else:
        if not metadata["has_audio"]:
            raise ValueError("上传文件不含音频流")
        metadata["duration"] = metadata.get("audio_duration") or metadata["duration"]
    return metadata


def load_image(item: dict[str, Any]) -> torch.Tensor:
    with Image.open(resolve_media_path(str(item["path"]))) as image:
        if image.width <= 0 or image.height <= 0 or image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError(f"图片尺寸过大或无效：{image.width}×{image.height}")
        oriented = ImageOps.exif_transpose(image)
        array = np.asarray(oriented.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def validate_image_tensor(value: Any, label: str, *, min_frames: int = 1) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 4:
        raise ValueError(f"{label} 必须是 ComfyUI IMAGE [B,H,W,C]")
    if (
        int(value.shape[0]) < min_frames
        or int(value.shape[1]) <= 0
        or int(value.shape[2]) <= 0
        or int(value.shape[3]) < 3
    ):
        raise ValueError(f"{label} 尺寸无效：{tuple(value.shape)}")
    return value


def official_reference_frame_count(input_frames: int, output_frames: int) -> int:
    count = min(int(input_frames), int(output_frames))
    if count < 5:
        return count
    return count - ((count - 5) % 17)


def stream_start_seconds(stream: Any) -> float:
    if stream is None or stream.start_time is None or stream.time_base is None:
        return 0.0
    value = float(stream.start_time * stream.time_base)
    return value if np.isfinite(value) else 0.0


def _stream_duration_seconds(stream: Any) -> float:
    if stream is None or stream.duration is None or stream.time_base is None:
        return 0.0
    return positive_float(stream.duration * stream.time_base)


def _audio_frame_start(frame: Any, stream_start: float, fallback: float) -> float:
    if frame.pts is not None and frame.time_base is not None:
        value = float(frame.pts * frame.time_base) - stream_start
        if np.isfinite(value):
            return value
    if frame.time is not None:
        value = float(frame.time) - stream_start
        if np.isfinite(value):
            return value
    return fallback


def _decode_audio_interval(
    container: Any,
    stream: Any,
    source_start: float,
    source_end: float,
    sample_rate: int,
) -> torch.Tensor:
    """Decode only the source interval instead of materializing the full file."""
    if source_end <= source_start:
        return torch.empty((1, 0), dtype=torch.float32)
    time_base = float(stream.time_base) if stream.time_base is not None else 0.0
    stream_start = stream_start_seconds(stream)
    if time_base > 0 and source_start > 0:
        seek_seconds = max(0.0, source_start - 1.0)
        seek_pts = int(stream.start_time or 0) + int(seek_seconds / time_base)
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

    resampler = av.AudioResampler(format="fltp")
    pieces: list[torch.Tensor] = []
    fallback = max(0.0, source_start - 1.0)
    stop = False

    def consume(frame: Any) -> None:
        nonlocal fallback, stop
        if stop:
            return
        output_rate = int(frame.sample_rate or sample_rate)
        if output_rate != sample_rate:
            raise ValueError(f"音频采样率在解码过程中变化：{sample_rate} → {output_rate}")
        array = frame.to_ndarray()
        if array.ndim == 1:
            array = array[None, :]
        frame_start = _audio_frame_start(frame, stream_start, fallback)
        frame_samples = int(array.shape[-1])
        frame_end = frame_start + frame_samples / sample_rate
        fallback = frame_end
        if frame_end <= source_start + 1e-9:
            return
        if frame_start >= source_end - 1e-9:
            stop = True
            return
        left = max(0, round((source_start - frame_start) * sample_rate))
        right = min(frame_samples, round((source_end - frame_start) * sample_rate))
        if right > left:
            pieces.append(torch.from_numpy(np.ascontiguousarray(array[:, left:right])).to(torch.float32))
        if frame_end >= source_end - 1e-9:
            stop = True

    for decoded in container.decode(stream):
        for frame in resampler.resample(decoded):
            consume(frame)
            if stop:
                break
        if stop:
            break
    if not stop:
        for frame in resampler.resample(None):
            consume(frame)
            if stop:
                break
    if not pieces:
        return torch.empty((1, 0), dtype=torch.float32)
    return torch.cat(pieces, dim=-1)


def load_audio(item: dict[str, Any]) -> tuple[dict[str, Any], float]:
    path = resolve_media_path(str(item["path"]))
    align_to_video = bool(item.get("align_to_video"))
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"媒体文件不含音轨：{item['path']}")
        stream = container.streams.audio[0]
        sample_rate = int(stream.sample_rate or 0)
        if sample_rate <= 0 or sample_rate > MAX_AUDIO_SAMPLE_RATE:
            raise ValueError(f"音频采样率无效或过高：{sample_rate} Hz（{item['path']}）")

        metadata_duration = positive_float(item.get("audio_duration")) or positive_float(item.get("duration"))
        audio_duration = metadata_duration or _stream_duration_seconds(stream)
        if audio_duration <= 0 and container.duration is not None:
            audio_duration = positive_float(container.duration / av.time_base)
        if audio_duration <= 0:
            raise ValueError(f"无法确定音频时长：{item['path']}")

        start_value = item.get("trim_start", 0.0)
        requested_start = max(0.0, finite_float(0.0 if start_value in (None, "") else start_value, "音频裁剪入点"))
        end_value = item.get("trim_end", 0.0)
        raw_end = 0.0 if end_value in (None, "") else finite_float(end_value, "音频裁剪出点")
        timeline_duration = positive_float(item.get("timeline_duration")) or audio_duration
        requested_end = timeline_duration if raw_end <= 0 else min(max(raw_end, 0.0), timeline_duration)
        if requested_end <= requested_start:
            raise ValueError(f"无效音频裁剪区间：{requested_start:.3f}s–{requested_end:.3f}s")
        target_duration = requested_end - requested_start
        validate_reference_audio_duration(target_duration, "每段参考音频")

        offset = 0.0
        if align_to_video:
            if not container.streams.video:
                raise ValueError(f"无法按视频时间轴对齐：媒体文件不含视频流：{item['path']}")
            offset = stream_start_seconds(container.streams.video[0]) - stream_start_seconds(stream)
        source_start = requested_start + offset
        source_end = requested_end + offset
        overlap_start = max(0.0, source_start)
        overlap_end = min(audio_duration, source_end)
        decoded = _decode_audio_interval(container, stream, overlap_start, overlap_end, sample_rate)

    if align_to_video:
        target_samples = round(target_duration * sample_rate)
        left_pad = min(target_samples, max(0, round(-source_start * sample_rate)))
        available = decoded.shape[-1]
        right_pad = max(0, target_samples - left_pad - available)
        waveform = F.pad(decoded, (left_pad, right_pad))[..., :target_samples]
    else:
        expected = max(0, round((overlap_end - overlap_start) * sample_rate))
        if decoded.shape[-1] < expected and expected - decoded.shape[-1] <= 2:
            decoded = F.pad(decoded, (0, expected - decoded.shape[-1]))
        waveform = decoded[..., :expected]
    if waveform.shape[-1] <= 0:
        raise ValueError(f"音频没有可解码采样：{item['path']}")
    normalized = normalize_audio_for_h3({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate})
    return normalized, normalized["waveform"].shape[-1] / sample_rate


def _reference_decode_dimensions(stream: Any, frame: Any, rotation_quarters: int) -> tuple[int, int]:
    raw_width = int(getattr(stream, "width", 0) or getattr(frame, "width", 0) or 0)
    raw_height = int(getattr(stream, "height", 0) or getattr(frame, "height", 0) or 0)
    if raw_width <= 0 or raw_height <= 0:
        raise ValueError("视频帧宽高无效")
    sar = positive_float(getattr(stream, "sample_aspect_ratio", None)) or 1.0
    if not 0.05 <= sar <= 20.0:
        sar = 1.0
    display_width = max(1, round(raw_width * sar))
    display_height = raw_height
    if rotation_quarters % 2:
        display_width, display_height = display_height, display_width
    target_width, target_height = adapt_canvas(display_width, display_height)
    if display_width * display_height < target_width * target_height:
        target_width = max(CANVAS_MULTIPLE, round(display_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        target_height = max(CANVAS_MULTIPLE, round(display_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return (target_height, target_width) if rotation_quarters % 2 else (target_width, target_height)


def _decode_video_rgb24(path: Path, start: float, end: float) -> torch.Tensor:
    start = finite_float(start, "视频裁剪入点")
    end = finite_float(end, "视频裁剪出点")
    if end <= start:
        raise ValueError(f"无效视频裁剪区间：{start:.3f}s–{end:.3f}s")
    target_count = max(1, round((end - start) * FPS))
    result: torch.Tensor | None = None
    target_index = 0
    previous: tuple[float, np.ndarray] | None = None

    def target_time(index: int) -> float:
        return start + index / FPS

    def store(image: np.ndarray) -> None:
        nonlocal result, target_index
        image = np.ascontiguousarray(image)
        if result is None:
            result = torch.empty((target_count, *image.shape), dtype=torch.float32)
        elif tuple(result.shape[1:]) != tuple(image.shape):
            raise ValueError(
                f"视频帧尺寸在解码过程中发生变化：{tuple(result.shape[1:])} → {tuple(image.shape)}"
            )
        result[target_index].copy_(torch.from_numpy(image).to(torch.float32).div_(255.0))
        target_index += 1

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"媒体文件不含视频流：{path.name}")
        stream = container.streams.video[0]
        time_base = float(stream.time_base) if stream.time_base is not None else 0.0
        stream_start_pts = int(stream.start_time or 0)
        stream_start = stream_start_pts * time_base if time_base > 0 else 0.0
        source_rate = (
            positive_float(getattr(stream, "average_rate", None))
            or positive_float(getattr(stream, "guessed_rate", None))
            or FPS
        )
        if time_base > 0 and start > 0:
            container.seek(
                stream_start_pts + int(start / time_base),
                stream=stream,
                backward=True,
                any_frame=False,
            )
        fallback_index = 0
        for frame in container.decode(stream):
            if frame.pts is not None and stream.time_base is not None:
                timestamp = float(frame.pts * stream.time_base) - stream_start
            elif frame.time is not None:
                timestamp = float(frame.time) - stream_start
            else:
                timestamp = start + fallback_index / source_rate
                fallback_index += 1
            if not np.isfinite(timestamp) or timestamp + 1e-6 < start:
                continue
            if timestamp >= end - 1e-6:
                break
            rotation = int(round(float(getattr(frame, "rotation", 0) or 0) / 90.0)) % 4
            decode_width, decode_height = _reference_decode_dimensions(stream, frame, rotation)
            image = frame.to_ndarray(format="rgb24", width=decode_width, height=decode_height)
            if rotation:
                image = np.rot90(image, k=rotation, axes=(0, 1)).copy()
            while target_index < target_count and target_time(target_index) <= timestamp + 1e-9:
                if (
                    previous is not None
                    and abs(previous[0] - target_time(target_index))
                    <= abs(timestamp - target_time(target_index))
                ):
                    store(previous[1])
                else:
                    store(image)
            previous = (timestamp, image)
            if target_index >= target_count:
                break

    if previous is None:
        raise ValueError(f"视频没有可解码帧：{path.name}")
    while target_index < target_count:
        store(previous[1])
    if result is None:
        raise RuntimeError("视频解码结果为空")
    return result


def load_video(
    item: dict[str, Any],
    output_frames: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any] | None, float, int]:
    path = resolve_media_path(str(item["path"]))
    video_duration = positive_float(item.get("video_duration")) or positive_float(item.get("duration"))
    metadata: dict[str, Any] | None = None
    if video_duration <= 0:
        metadata = probe_media(path)
        video_duration = float(metadata.get("video_duration") or metadata["duration"])
    start_value = item.get("trim_start", 0.0)
    start = min(
        max(
            finite_float(0.0 if start_value in (None, "") else start_value, "视频裁剪入点"),
            0.0,
        ),
        video_duration,
    )
    end_value = item.get("trim_end", 0.0)
    raw_end = 0.0 if end_value in (None, "") else finite_float(end_value, "视频裁剪出点")
    end = video_duration if raw_end <= 0 else min(max(raw_end, 0.0), video_duration)
    if end <= start:
        raise ValueError(f"无效视频裁剪区间：{start:.3f}s–{end:.3f}s")
    clip_duration = end - start
    validate_reference_duration(clip_duration, "每段参考视频")

    selected_frame_count = max(1, round(clip_duration * FPS))
    decoded_frame_count = (
        official_reference_frame_count(selected_frame_count, output_frames)
        if output_frames is not None
        else selected_frame_count
    )
    if decoded_frame_count < 5:
        raise ValueError("MiniMax H3 参考视频有效帧数不足 5 帧")
    frames = _decode_video_rgb24(path, start, min(end, start + decoded_frame_count / FPS))

    soundtrack = None
    if item.get("use_audio") and item.get("has_audio") is not False:
        # Silent uploads persist has_audio=False. Treat stale checked state as
        # disabled rather than decoding the video and then failing on a missing
        # audio stream. Older workflows without this metadata are still probed.
        audio_duration = positive_float(item.get("audio_duration"))
        if audio_duration <= 0:
            metadata = metadata or probe_media(path)
            audio_duration = positive_float(metadata.get("audio_duration"))
        soundtrack, _ = load_audio(
            {
                "path": item["path"],
                "trim_start": start,
                "trim_end": end,
                "align_to_video": True,
                "timeline_duration": video_duration,
                "audio_duration": audio_duration,
            }
        )
    return frames, soundtrack, clip_duration, selected_frame_count


def validate_reference_limits(
    images: dict[str, Any],
    videos: dict[str, Any],
    paired_audio: dict[str, Any],
    audios: dict[str, Any],
    video_durations: list[float],
    audio_durations: list[float],
    embedded_audio_count: int = 0,
) -> None:
    if len(images) > 9 or len(videos) > 3 or len(audios) > 3 or len(paired_audio) > 3:
        raise ValueError("Ref2VA 最多支持 9 张图片、3 段视频、3 段配对音频和 3 段独立音频")
    if not images and not videos and audios:
        raise ValueError("独立音频不能作为唯一参考；请至少提供图片或视频")
    expected_paired = {name.replace("ref_video_", "ref_video_audio_") for name in videos}
    orphan = set(paired_audio) - expected_paired
    if orphan:
        raise ValueError(f"配对音频缺少同编号视频：{', '.join(sorted(orphan))}")
    if any(not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS for duration in video_durations):
        raise ValueError("每段参考视频必须为 2–15 秒")
    if sum(video_durations) > MAX_REFERENCE_SECONDS + 1e-6:
        raise ValueError("参考视频总时长不能超过 15 秒")
    for duration in audio_durations:
        validate_reference_audio_duration(duration)
    file_count = len(images) + len(videos) + len(audios) + len(paired_audio) - embedded_audio_count
    if file_count > 12:
        raise ValueError("Ref2VA 混合参考文件总数不能超过 12")
