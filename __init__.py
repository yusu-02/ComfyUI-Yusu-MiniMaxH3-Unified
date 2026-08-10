from __future__ import annotations

import asyncio
import logging

from aiohttp import web
from server import PromptServer

from .media import (
    MAX_UPLOAD_BYTES,
    resolve_media_path,
    validate_upload_name,
    validate_uploaded_file,
)
from .nodes import comfy_entrypoint

WEB_DIRECTORY = "./web"
LOGGER = logging.getLogger(__name__)


async def inspect_media(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("媒体检查请求必须是 JSON 对象")
        relative_path = str(body.get("path", ""))
        expected_kind = str(body.get("kind", ""))
        mime = str(body.get("mime", ""))
        path = resolve_media_path(relative_path)
        size = path.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            return web.json_response({"error": "文件超过 512 MiB 上限"}, status=413)
        _extension, kind = validate_upload_name(path.name, mime)
        if expected_kind != kind:
            raise ValueError(f"该槽位需要 {expected_kind or '未知'}，实际上传的是 {kind}")
        metadata = await asyncio.to_thread(validate_uploaded_file, path, kind)
        return web.json_response(
            {
                "path": relative_path,
                "kind": kind,
                "size": size,
                **metadata,
            }
        )
    except (ValueError, OSError, web.HTTPException) as error:
        return web.json_response({"error": str(error)}, status=400)
    except Exception:
        LOGGER.exception("MiniMax H3 媒体检查发生未预期错误")
        return web.json_response({"error": "媒体检查失败，请查看 ComfyUI 控制台日志"}, status=500)


server_instance = getattr(PromptServer, "instance", None)
if server_instance is not None:
    server_instance.routes.post("/minimax_h3_unified/inspect")(inspect_media)


__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
