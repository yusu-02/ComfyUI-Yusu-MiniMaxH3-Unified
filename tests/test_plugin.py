from __future__ import annotations

import importlib
import json
import shutil
import struct
import sys
import tempfile
import types
import unittest
import wave
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


class _Factory:
    @classmethod
    def Input(cls, name, **kwargs):
        return {"kind": cls.__name__, "name": name, **kwargs}

    @classmethod
    def Output(cls, **kwargs):
        return {"kind": cls.__name__, **kwargs}


class _DynamicCombo(_Factory):
    @staticmethod
    def Option(name, inputs):
        return {"name": name, "inputs": inputs}


class _Autogrow(_Factory):
    Type = dict

    @staticmethod
    def TemplatePrefix(**kwargs):
        return kwargs


class _Custom:
    def __init__(self, name):
        self.name = name

    def Output(self, **kwargs):
        return {"kind": self.name, **kwargs}


def _install_import_stubs() -> None:
    av = types.ModuleType("av")
    av.time_base = 1_000_000
    av.open = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("PyAV test stub"))
    av.AudioResampler = object
    sys.modules["av"] = av

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_input_directory = lambda: str(PLUGIN_ROOT / "tests" / "runtime_tmp")
    folder_paths.get_annotated_filepath = lambda name: str(Path(folder_paths.get_input_directory()) / name)
    sys.modules["folder_paths"] = folder_paths

    io = types.SimpleNamespace(
        ComfyNode=type("ComfyNode", (), {}),
        Autogrow=_Autogrow,
        DynamicCombo=_DynamicCombo,
        Image=type("Image", (_Factory,), {}),
        Audio=type("Audio", (_Factory,), {}),
        Vae=type("Vae", (_Factory,), {}),
        Combo=type("Combo", (_Factory,), {}),
        Model=type("Model", (_Factory,), {}),
        Clip=type("Clip", (_Factory,), {}),
        String=type("String", (_Factory,), {}),
        Int=type("Int", (_Factory,), {}),
        Boolean=type("Boolean", (_Factory,), {}),
        Conditioning=type("Conditioning", (_Factory,), {}),
        Latent=type("Latent", (_Factory,), {}),
        Float=type("Float", (_Factory,), {}),
        Custom=_Custom,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=lambda *args: args,
    )
    latest = types.ModuleType("comfy_api.latest")
    latest.ComfyExtension = type("ComfyExtension", (), {})
    latest.io = io
    sys.modules["comfy_api"] = types.ModuleType("comfy_api")
    sys.modules["comfy_api.latest"] = latest

    official = types.ModuleType("comfy_extras.nodes_minimax_h3")
    official.FPS = 24
    official.adapt_canvas = lambda width, height: (
        max(32, round(width / 32) * 32),
        max(32, round(height / 32) * 32),
    )
    def temporal_shape(length):
        frame_count = max(5, int(length))
        while frame_count % 17 != 5:
            frame_count += 1
        video_latent_t = 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2
        audio_latent_t = round((frame_count / 24) * 40)
        return frame_count, video_latent_t, audio_latent_t

    official.temporal_shape = temporal_shape

    class ImageNode:
        @staticmethod
        def execute(*_args):
            return [[torch.zeros(1), {}]], {"samples": torch.zeros(1)}

    class ReferenceNode:
        last_call = None

        @classmethod
        def execute(
            cls,
            clip,
            vae,
            audio_vae,
            prompt,
            width,
            height,
            length,
            ref_image_size,
            images,
            videos,
            paired_audio,
            audios,
        ):
            cls.last_call = {
                "clip": clip,
                "vae": vae,
                "audio_vae": audio_vae,
                "prompt": prompt,
                "length": length,
                "images": images,
                "videos": videos,
                "paired_audio": paired_audio,
                "audios": audios,
            }
            blocks = []
            for video_slot in videos:
                audio_slot = video_slot.replace("ref_video_", "ref_video_audio_")
                if audio_slot in paired_audio:
                    blocks.append({"audio_latent": torch.zeros(1, 32, 2, 5), "ref_audio_t": 5})
                blocks.append({"video_latent": torch.zeros(1)})
            for _slot in audios:
                blocks.append({"audio_latent": torch.zeros(1, 32, 2, 5), "ref_audio_t": 5})
            return [[torch.zeros(1), {"minimax_refs": blocks}]], {"samples": torch.zeros(1)}

    official.MiniMaxH3ImageToVideo = ImageNode
    official.MiniMaxH3ReferenceToVideo = ReferenceNode
    sys.modules["comfy_extras"] = types.ModuleType("comfy_extras")
    sys.modules["comfy_extras.nodes_minimax_h3"] = official

    package = types.ModuleType("h3u_plugin")
    package.__path__ = [str(PLUGIN_ROOT)]
    sys.modules["h3u_plugin"] = package


_install_import_stubs()
media = importlib.import_module("h3u_plugin.media")
nodes = importlib.import_module("h3u_plugin.nodes")
official = sys.modules["comfy_extras.nodes_minimax_h3"]


def audio(seconds: float = 2.0, sample_rate: int = 32000, channels: int = 1, value: float = 0.25):
    samples = round(seconds * sample_rate)
    return {
        "waveform": torch.full((1, channels, samples), value, dtype=torch.float32),
        "sample_rate": sample_rate,
    }


class PluginTests(unittest.TestCase):
    def test_public_duration_input_is_float_and_not_length(self):
        schema = nodes.MiniMaxH3Unified.define_schema()
        inputs = schema["inputs"]
        duration_inputs = [item for item in inputs if item.get("name") == "duration"]
        self.assertEqual(len(duration_inputs), 1)
        self.assertEqual(duration_inputs[0]["kind"], "Float")
        self.assertEqual(duration_inputs[0]["display_name"], "duration")
        self.assertFalse(any(item.get("name") == "length" for item in inputs))


    def test_schema_has_persistent_codecs_and_dynamic_media_inputs(self):
        schema = nodes.MiniMaxH3Unified.define_schema()
        inputs = schema["inputs"]
        names = [item.get("name") for item in inputs]
        mode = next(item for item in inputs if item.get("name") == "mode")
        self.assertIn("DynamicCombo", mode["kind"])
        self.assertNotIn("model", names)
        self.assertNotIn("fl2va_model", names)
        self.assertNotIn("ref2va_model", names)
        for name in ("clip", "vae", "audio_vae"):
            self.assertIn(name, names)
        self.assertNotIn("first_frame", names)
        serialized = repr(mode)
        for name in ("first_frame", "last_frame", "ref_image_", "ref_video_", "ref_video_audio_", "ref_audio_"):
            self.assertIn(name, serialized)
        self.assertNotIn('Input("audio_vae"', serialized)

    def tearDown(self):
        runtime = PLUGIN_ROOT / "tests" / "runtime_tmp"
        if runtime.exists():
            import shutil

            shutil.rmtree(runtime)

    def test_numeric_validation_rejects_non_finite_values(self):
        self.assertEqual(media.finite_float("1.25", "值"), 1.25)
        with self.assertRaisesRegex(ValueError, "有限数字"):
            media.finite_float(float("nan"), "值")
        with self.assertRaisesRegex(ValueError, "有限数字"):
            media.finite_float(None, "值")

    def test_audio_normalization_preserves_official_channels_and_values(self):
        normalized = media.normalize_audio_for_h3(audio(2.0, channels=1, value=1.25))
        self.assertEqual(normalized["waveform"].shape, (1, 1, 64000))
        self.assertTrue(torch.all(normalized["waveform"] == 1.25))
        with self.assertRaisesRegex(ValueError, "静音"):
            media.normalize_audio_for_h3(audio(value=0.0))
        bad = audio()
        bad["waveform"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN"):
            media.normalize_audio_for_h3(bad)

    def test_sparse_autogrow_mapping_does_not_renumber(self):
        marker0, marker2 = object(), object()
        resolved = nodes.autogrow_slots(
            "ref_audio",
            {
                "mode.ref_audios.ref_audio_0": marker0,
                "mode.ref_audios.ref_audio_2": marker2,
            },
        )
        self.assertIs(resolved["ref_audio_1"], marker0)
        self.assertIs(resolved["ref_audio_3"], marker2)
        self.assertNotIn("ref_audio_2", resolved)

    def test_external_slot_only_overrides_same_internal_slot(self):
        state = {
            "ref_image_1": {"path": "one"},
            "ref_image_2": {"path": "two"},
            "ref_image_3": {"path": "three"},
        }
        replacement = object()
        resolved = nodes.resolve_slots("ref_image", 9, state, {"ref_image_2": replacement})
        self.assertEqual(resolved["ref_image_1"], state["ref_image_1"])
        self.assertIs(resolved["ref_image_2"], replacement)
        self.assertEqual(resolved["ref_image_3"], state["ref_image_3"])

    def test_media_state_and_path_security(self):
        state = {"ref_audio_1": {"path": "minimax_h3_unified/a.wav"}}
        self.assertEqual(nodes.parse_media_state(json.dumps(state)), state)
        with self.assertRaisesRegex(ValueError, "不是有效 JSON"):
            nodes.parse_media_state("{bad")
        with self.assertRaisesRegex(ValueError, "必须是 JSON 对象"):
            nodes.parse_media_state("[]")
        with self.assertRaisesRegex(ValueError, "非法媒体路径"):
            media.resolve_media_path("../secret.wav")
        for name in ("../bad.wav", "folder/bad.wav", "bad.exe"):
            with self.assertRaises(ValueError):
                media.validate_upload_name(name, "audio/wav")
        self.assertEqual(media.validate_upload_name("ok.flac", "application/octet-stream"), (".flac", "audio"))

    def test_media_path_uses_comfyui_resolver(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "api" / "a.wav"
            path.parent.mkdir()
            path.touch()
            with patch.object(media.folder_paths, "get_annotated_filepath", return_value=str(path)) as resolve:
                self.assertEqual(media.resolve_media_path("api/a.wav"), path.resolve())
                resolve.assert_called_once_with("api/a.wav")

    def test_generation_and_reference_limits(self):
        media.validate_generation_size(1344, 768)
        with self.assertRaisesRegex(ValueError, "32 的倍数"):
            media.validate_generation_size(1920, 1080)
        self.assertEqual(media.official_reference_frame_count(362, 124), 124)
        self.assertEqual(media.official_reference_frame_count(120, 120), 107)
        media.validate_reference_limits({"a": 1}, {}, {}, {}, [], [])
        media.validate_reference_limits(
            {"a": 1},
            {},
            {},
            {"ref_audio_1": audio(10), "ref_audio_2": audio(10)},
            [],
            [10, 10],
        )
        with self.assertRaisesRegex(ValueError, "2–15"):
            media.validate_reference_limits({"a": 1}, {}, {}, {"ref_audio_1": audio(15.001)}, [], [15.001])
        with self.assertRaisesRegex(ValueError, "唯一参考"):
            media.validate_reference_limits({}, {}, {}, {"ref_audio_1": audio()}, [], [2.0])
        with self.assertRaisesRegex(ValueError, "同编号视频"):
            media.validate_reference_limits({"a": 1}, {}, {"ref_video_audio_1": audio()}, {}, [], [2.0])

    def test_image_loader_applies_exif_orientation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / media.MEDIA_SUBDIR
            root.mkdir()
            path = root / "oriented.jpg"
            source = Image.new("RGB", (2, 3), "red")
            exif = source.getexif()
            exif[274] = 6
            source.save(path, exif=exif)
            with patch.object(media.folder_paths, "get_input_directory", return_value=temp):
                loaded = media.load_image({"path": f"{media.MEDIA_SUBDIR}/oriented.jpg"})
            self.assertEqual(loaded.shape, (1, 2, 3, 3))

    def test_ffprobe_validates_real_wav_without_pyav(self):
        if shutil.which("ffprobe") is None:
            self.skipTest("ffprobe is not installed")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(8000)
                output.writeframes(b"".join(struct.pack("<hh", 100, -100) for _ in range(16000)))
            metadata = media.validate_uploaded_file(path, "audio")
            self.assertTrue(metadata["has_audio"])
            self.assertAlmostEqual(metadata["duration"], 2.0, places=2)

    def test_internal_audio_decoder_seeks_and_stops_at_trim_end(self):
        sample_rate = 10

        class Frame:
            def __init__(self, second):
                self.pts = second
                self.time_base = Fraction(1, 1)
                self.time = None
                self.sample_rate = sample_rate
                self._array = np.full((1, sample_rate), 0.5, dtype=np.float32)

            def to_ndarray(self):
                return self._array

        class PassResampler:
            def __init__(self, **_kwargs):
                pass

            def resample(self, frame):
                return [] if frame is None else [frame]

        stream = types.SimpleNamespace(
            sample_rate=sample_rate,
            start_time=0,
            time_base=Fraction(1, 1),
            duration=20,
            type="audio",
        )

        class Streams(list):
            @property
            def audio(self):
                return [stream]

            @property
            def video(self):
                return []

        class Container:
            duration = 20_000_000

            def __init__(self):
                self.streams = Streams([stream])
                self.seek_calls = []
                self.decoded = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def seek(self, *args, **kwargs):
                self.seek_calls.append((args, kwargs))

            def decode(self, _stream):
                for second in range(20):
                    self.decoded += 1
                    yield Frame(second)

        container = Container()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / media.MEDIA_SUBDIR
            root.mkdir()
            path = root / "audio.wav"
            path.write_bytes(b"stub")
            with (
                patch.object(media.folder_paths, "get_input_directory", return_value=temp),
                patch.object(media.av, "open", return_value=container),
                patch.object(media.av, "AudioResampler", PassResampler),
            ):
                loaded, duration = media.load_audio(
                    {
                        "path": f"{media.MEDIA_SUBDIR}/audio.wav",
                        "duration": 20,
                        "audio_duration": 20,
                        "trim_start": 5,
                        "trim_end": 7,
                    }
                )
        self.assertTrue(container.seek_calls)
        self.assertLess(container.decoded, 20)
        self.assertEqual(loaded["waveform"].shape, (1, 1, 20))
        self.assertEqual(duration, 2.0)

    def test_embedded_audio_alignment_pads_late_audio_start(self):
        sample_rate = 10

        class Frame:
            def __init__(self, pts):
                self.pts = pts
                self.time_base = Fraction(1, 10)
                self.time = None
                self.sample_rate = sample_rate

            def to_ndarray(self):
                return np.full((1, sample_rate), 0.5, dtype=np.float32)

        class PassResampler:
            def __init__(self, **_kwargs):
                pass

            def resample(self, frame):
                return [] if frame is None else [frame]

        audio_stream = types.SimpleNamespace(
            sample_rate=sample_rate,
            start_time=2,
            time_base=Fraction(1, 10),
            duration=200,
            type="audio",
        )
        video_stream = types.SimpleNamespace(
            start_time=0,
            time_base=Fraction(1, 10),
            duration=200,
            type="video",
        )

        class Streams(list):
            @property
            def audio(self):
                return [audio_stream]

            @property
            def video(self):
                return [video_stream]

        class Container:
            duration = 20_000_000

            def __init__(self):
                self.streams = Streams([video_stream, audio_stream])

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def seek(self, *_args, **_kwargs):
                pass

            def decode(self, _stream):
                for second in range(20):
                    yield Frame(2 + second * 10)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / media.MEDIA_SUBDIR
            root.mkdir()
            path = root / "video.mp4"
            path.write_bytes(b"stub")
            with (
                patch.object(media.folder_paths, "get_input_directory", return_value=temp),
                patch.object(media.av, "open", return_value=Container()),
                patch.object(media.av, "AudioResampler", PassResampler),
            ):
                loaded, duration = media.load_audio(
                    {
                        "path": f"{media.MEDIA_SUBDIR}/video.mp4",
                        "audio_duration": 20,
                        "timeline_duration": 20,
                        "trim_start": 0,
                        "trim_end": 2,
                        "align_to_video": True,
                    }
                )
        self.assertEqual(duration, 2.0)
        self.assertEqual(loaded["waveform"].shape, (1, 1, 20))
        self.assertTrue(torch.equal(loaded["waveform"][..., :2], torch.zeros(1, 1, 2)))
        self.assertTrue(torch.all(loaded["waveform"][..., 2:] == 0.5))

    def test_omni_execute_passes_audio_to_official_node(self):
        image = torch.zeros(1, 32, 32, 3)
        result = nodes.MiniMaxH3Unified.execute(
            mode="omni_reference",
            clip=object(),
            vae=object(),
            audio_vae=object(),
            prompt="use <Picture 1> and <Audio 1>",
            width=1344,
            height=768,
            duration=124 / 24,
            ref_images={"mode.ref_images.ref_image_0": image},
            ref_audios={"mode.ref_audios.ref_audio_0": audio(2.0)},
        )
        self.assertIn("ref_audio_1", official.MiniMaxH3ReferenceToVideo.last_call["audios"])
        self.assertIsInstance(result[2], dict)
        self.assertEqual(result[2]["waveform"].shape, (1, 1, 64000))
        self.assertEqual(result[2]["sample_rate"], 32000)

    def test_auto_length_uses_longest_reference_audio_and_official_grid(self):
        image = torch.zeros(1, 32, 32, 3)
        result = nodes.MiniMaxH3Unified.execute(
            mode="omni_reference",
            clip=object(),
            vae=object(),
            audio_vae=object(),
            prompt="use <Picture 1>, <Audio 1> and <Audio 2>",
            width=1344,
            height=768,
            duration=124 / 24,
            auto_length_from_audio=True,
            ref_images={"mode.ref_images.ref_image_0": image},
            ref_audios={
                "mode.ref_audios.ref_audio_0": audio(2.0),
                "mode.ref_audios.ref_audio_1": audio(5.817),
            },
        )
        self.assertEqual(official.MiniMaxH3ReferenceToVideo.last_call["length"], 141)
        self.assertEqual(result[2]["waveform"].shape[-1], round(5.817 * 32000))
        self.assertEqual(len(result), 3)

    def test_auto_length_without_audio_falls_back_to_manual_length(self):
        image = torch.zeros(1, 32, 32, 3)
        result = nodes.MiniMaxH3Unified.execute(
            mode="omni_reference",
            clip=object(),
            vae=object(),
            audio_vae=None,
            prompt="use <Picture 1>",
            width=1344,
            height=768,
            duration=124 / 24,
            auto_length_from_audio=True,
            ref_images={"mode.ref_images.ref_image_0": image},
        )
        self.assertEqual(official.MiniMaxH3ReferenceToVideo.last_call["length"], 124)
        self.assertIsNone(result[2])
        self.assertEqual(len(result), 3)

    def test_non_audio_modes_always_use_manual_length(self):
        for mode in ("text_to_video", "first_last_frame"):
            kwargs = {}
            if mode == "first_last_frame":
                kwargs["first_frame"] = torch.zeros(1, 32, 32, 3)
            result = nodes.MiniMaxH3Unified.execute(
                mode=mode,
                    clip=object(),
                vae=object(),
                prompt="test",
                width=1344,
                height=768,
                duration=124 / 24,
                auto_length_from_audio=True,
                **kwargs,
            )
            self.assertIsNone(result[2])
            self.assertEqual(len(result), 3)

    def test_embedded_audio_duration_can_drive_auto_length_without_loading_twice(self):
        candidates = nodes._embedded_audio_duration_candidates(
            {
                "ref_video_1": {
                    "duration": 10.0,
                    "video_duration": 10.0,
                    "trim_start": 1.0,
                    "trim_end": 6.817,
                    "use_audio": True,
                }
            },
            {},
        )
        self.assertEqual(candidates, [5.817])
        shape = nodes._resolve_frame_count(124, True, candidates)
        self.assertEqual(shape, 141)


    def test_silent_video_is_not_treated_as_reference_audio(self):
        silent = {
            "ref_video_1": {
                "duration": 5.817,
                "video_duration": 5.817,
                "trim_start": 0.0,
                "trim_end": 5.817,
                "use_audio": True,
                "has_audio": False,
            }
        }
        self.assertEqual(nodes._embedded_audio_duration_candidates(silent, {}), [])

        legacy_unknown = {"ref_video_1": {**silent["ref_video_1"]}}
        legacy_unknown["ref_video_1"].pop("has_audio")
        self.assertEqual(
            nodes._embedded_audio_duration_candidates(legacy_unknown, {}),
            [5.817],
        )


    def test_silent_video_stale_checkbox_does_not_decode_audio(self):
        item = {
            "path": "minimax_h3_unified/silent.mp4",
            "duration": 5.0,
            "video_duration": 5.0,
            "trim_start": 0.0,
            "trim_end": 5.0,
            "use_audio": True,
            "has_audio": False,
        }
        with (
            patch.object(media, "resolve_media_path", return_value=Path("/tmp/silent.mp4")),
            patch.object(
                media,
                "_decode_video_rgb24",
                return_value=torch.zeros(107, 32, 32, 3),
            ),
            patch.object(media, "load_audio") as load_audio_mock,
        ):
            _frames, soundtrack, duration, selected_frames = media.load_video(item, 124)
        self.assertIsNone(soundtrack)
        self.assertAlmostEqual(duration, 5.0)
        self.assertEqual(selected_frames, 120)
        load_audio_mock.assert_not_called()

    def test_orphan_paired_audio_is_rejected_before_decode(self):
        image = torch.zeros(1, 32, 32, 3)
        with (
            patch.object(nodes, "_prepare_audio_slots") as prepare,
            self.assertRaisesRegex(ValueError, "同编号视频"),
        ):
            nodes.MiniMaxH3Unified.execute(
                mode="omni_reference",
                    clip=object(),
                vae=object(),
                audio_vae=object(),
                prompt="test",
                width=1344,
                height=768,
                duration=5.0,
                ref_images={"mode.ref_images.ref_image_0": image},
                ref_video_audios={"mode.ref_video_audios.ref_video_audio_0": audio(2.0)},
            )
        prepare.assert_not_called()

    def test_auto_length_matches_official_round_expression_at_boundary(self):
        # 5.88 * 24 = 141.12. Official round() keeps 141; ceil() would
        # incorrectly jump to the next 17k+5 grid value (158).
        frame_count = nodes._resolve_frame_count(124, True, [5.88])
        self.assertEqual(frame_count, 141)

    def test_manual_seconds_are_converted_and_aligned_to_h3_grid(self):
        self.assertEqual(nodes._resolve_frame_count(5.0, False, []), 124)
        self.assertEqual(nodes._resolve_frame_count(0, False, []), 5)
        self.assertEqual(nodes._resolve_frame_count(124, False, []), 2980)

        with self.assertRaisesRegex(ValueError, "不能小于 0"):
            nodes._resolve_frame_count(-0.1, False, [])

    def test_audio_output_selects_longest_reference_without_copying(self):
        short = audio(2.0)
        long = audio(5.0)
        selected = nodes._select_reference_audio([
            ("ref_audio_1", short, 2.0),
            ("ref_audio_2", long, 5.0),
        ])
        self.assertIs(selected, long)
        self.assertIsNone(nodes._select_reference_audio([]))

    def test_schema_preserves_official_socket_names(self):
        schema = nodes.MiniMaxH3Unified.define_schema()
        serialized = repr(schema)
        self.assertNotIn('"model"', repr(schema["outputs"]))
        for name in ("audio_vae", "ref_image_", "ref_video_", "ref_video_audio_", "ref_audio_"):
            self.assertIn(name, serialized)
        self.assertEqual([item["display_name"] for item in schema["outputs"]], [
            "positive", "av_latent", "audio"
        ])
        self.assertEqual(schema["outputs"][2]["kind"], "Audio")
        for removed in (
            "media_info",
            "encoded_reference_audio",
            "total_reference_audio_duration",
            "effective_length",
            "generation_duration",
            "length_source_audio_duration",
        ):
            self.assertNotIn(removed, repr(schema["outputs"]))
        auto_input = next(item for item in schema["inputs"] if item.get("name") == "auto_length_from_audio")
        self.assertFalse(auto_input["default"])
        duration_input = next(item for item in schema["inputs"] if item.get("name") == "duration")
        self.assertEqual(duration_input["kind"], "Float")
        self.assertEqual(duration_input["display_name"], "duration")
        self.assertEqual(duration_input["min"], 0.0)
        self.assertNotIn("max", duration_input)

    def test_release_has_no_background_polling_or_telemetry(self):
        js = (PLUGIN_ROOT / "web" / "minimax_h3_unified.js").read_text(encoding="utf-8")
        backend = (PLUGIN_ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("setInterval(", js)
        self.assertNotIn("WebSocket(", js)
        self.assertNotIn("sendBeacon(", js)
        self.assertNotIn("origin.serialize", js)
        self.assertIn("asyncio.to_thread(validate_uploaded_file", backend)
        self.assertNotIn('routes.post("/minimax_h3_unified/upload")', backend)
        self.assertIn("MAX_WAVEFORM_CACHE_ITEMS", js)
        self.assertIn('durationWidget.value === ""', js)
        self.assertIn("item.disabled = Boolean(disabled)", js)
        self.assertIn("timeline.setPointerCapture", js)
        self.assertIn("controls.onpointerdown", js)
        self.assertIn('api.fetchApi("/upload/image"', js)
        self.assertIn('api.fetchApi("/minimax_h3_unified/inspect"', js)
        self.assertNotIn("uploadResponse.json()", js)
        self.assertNotIn("inspectResponse.json()", js)
        self.assertIn("inspectResponse.status === 404 || inspectResponse.status === 405", js)
        self.assertIn("canvas.onpointerdown", js)
        self.assertIn("expandCollapsedAutogrowInputs", js)
        self.assertIn("afterResize: syncPanelWidth", js)
        self.assertIn("nodeWidth - 20", js)
        self.assertNotIn('api.fetchApi("/minimax_h3_unified/upload"', js)


if __name__ == "__main__":
    unittest.main()
