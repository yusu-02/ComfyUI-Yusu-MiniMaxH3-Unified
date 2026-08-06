from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from aiohttp import web
from server import PromptServer

from .media import (
    MAX_UPLOAD_BYTES,
    MEDIA_SUBDIR,
    media_root,
    validate_upload_name,
    validate_uploaded_file,
)
from .nodes import comfy_entrypoint

WEB_DIRECTORY = "./web"
LOGGER = logging.getLogger(__name__)
UPLOAD_CHUNK_BYTES = 1024 * 1024


def _remove_partial(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        LOGGER.warning("无法清理未完成上传：%s", path, exc_info=True)


async def upload_media(request: web.Request) -> web.Response:
    destination: Path | None = None
    try:
        if request.content_length is not None and request.content_length > MAX_UPLOAD_BYTES:
            return web.json_response({"error": "文件超过 512 MiB 上限"}, status=413)
        if not str(request.content_type or "").startswith("multipart/"):
            return web.json_response({"error": "上传请求必须使用 multipart/form-data"}, status=400)

        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "file" or not field.filename:
            return web.json_response({"error": "缺少 file 上传字段"}, status=400)

        extension, kind = validate_upload_name(field.filename, field.headers.get("Content-Type", ""))
        destination = media_root() / f"{uuid.uuid4().hex}{extension}"
        size = 0
        with destination.open("xb") as output:
            while True:
                chunk = await field.read_chunk(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise ValueError("文件超过 512 MiB 上限")
                output.write(chunk)

        metadata = await asyncio.to_thread(validate_uploaded_file, destination, kind)
        return web.json_response(
            {
                "path": f"{MEDIA_SUBDIR}/{destination.name}",
                "name": field.filename,
                "kind": kind,
                "size": size,
                **metadata,
            }
        )
    except asyncio.CancelledError:
        _remove_partial(destination)
        raise
    except (ValueError, OSError, web.HTTPException) as error:
        _remove_partial(destination)
        return web.json_response({"error": str(error)}, status=400)
    except Exception:
        _remove_partial(destination)
        LOGGER.exception("MiniMax H3 媒体上传发生未预期错误")
        return web.json_response({"error": "上传处理失败，请查看 ComfyUI 控制台日志"}, status=500)


server_instance = getattr(PromptServer, "instance", None)
if server_instance is not None:
    server_instance.routes.post("/minimax_h3_unified/upload")(upload_media)


__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
