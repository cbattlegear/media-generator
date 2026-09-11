"""Validated, bounded trailer uploads with atomic publication."""

import json
import os
import subprocess
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable
from uuid import uuid4

from fastapi import HTTPException, Request, Response, UploadFile
from fastapi.routing import APIRoute
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException
from starlette.staticfiles import StaticFiles
from starlette.types import Message, Receive, Scope, Send

TRAILERS_DIR = Path(__file__).resolve().parent.parent / "trailers"
PUBLIC_DIR = TRAILERS_DIR / "public"
STAGING_DIR = TRAILERS_DIR / ".uploads"
MAX_SIZE_MIB = int(os.getenv("TRAILER_MAX_SIZE_MIB", "250"))
if MAX_SIZE_MIB <= 0:
    raise ValueError("TRAILER_MAX_SIZE_MIB must be a positive integer")
MAX_SIZE_BYTES = MAX_SIZE_MIB * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 1024 * 1024


class TrailerStaticFiles(StaticFiles):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_response(message: Message) -> None:
            if message["type"] == "http.response.start" and message["status"] == 416:
                # Some supported Starlette versions omit the unit on unsatisfiable ranges.
                headers = MutableHeaders(scope=message)
                content_range = headers.get("content-range", "")
                if content_range.startswith("*/"):
                    headers["content-range"] = f"bytes {content_range}"
            await send(message)

        await super().__call__(scope, receive, send_response)


class TrailerUploadRoute(APIRoute):
    """Bound the request before multipart parsing can spool unlimited data."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def limited_handler(request: Request) -> Response:
            request_limit = MAX_SIZE_BYTES + MULTIPART_OVERHEAD_BYTES
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    length = int(content_length)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
                if length < 0:
                    raise HTTPException(status_code=400, detail="Invalid Content-Length")
                if length > request_limit:
                    raise HTTPException(
                        status_code=413, detail="Trailer upload request is too large"
                    )

            total = 0

            async def limited_receive() -> Message:
                nonlocal total
                message = await request.receive()
                if message["type"] == "http.request":
                    total += len(message.get("body", b""))
                    if total > request_limit:
                        # This exception makes Starlette close any spooled multipart files.
                        raise MultiPartException("Trailer upload request is too large")
                return message

            try:
                return await handler(Request(request.scope, receive=limited_receive))
            except StarletteHTTPException as exc:
                if total > request_limit:
                    raise HTTPException(
                        status_code=413, detail="Trailer upload request is too large"
                    ) from exc
                raise

        return limited_handler


def validate_mp4(path: Path) -> None:
    """Check the container and require a real video stream, not just cover art."""
    with path.open("rb") as source:
        header = source.read(16)
    if len(header) != 16 or header[4:8] != b"ftyp" or header[8:12] == b"qt  ":
        raise HTTPException(status_code=422, detail="File must contain an MP4 video")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-show_entries",
                "format=format_name:stream=codec_type,width,height:stream_disposition=attached_pic",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503, detail="Trailer validation requires ffprobe on the server"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=422, detail="Trailer validation timed out") from exc

    if result.returncode != 0:
        raise HTTPException(status_code=422, detail="File must contain a readable MP4 video")

    metadata = json.loads(result.stdout)
    formats = metadata.get("format", {}).get("format_name", "").split(",")
    has_video = any(
        stream.get("codec_type") == "video"
        and stream.get("width", 0) > 0
        and stream.get("height", 0) > 0
        and not stream.get("disposition", {}).get("attached_pic", 0)
        for stream in metadata.get("streams", [])
    )
    if "mp4" not in formats or not has_video:
        raise HTTPException(status_code=422, detail="File must contain an MP4 video stream")


@contextmanager
def store_trailer(file: UploadFile, movie_id: int) -> Iterator[str]:
    """Publish a unique file; remove it if the caller cannot commit its URL."""
    if file.content_type != "video/mp4":
        raise HTTPException(status_code=415, detail="Trailer must use the video/mp4 content type")
    if file.size is not None and file.size > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413, detail=f"Trailer exceeds the {MAX_SIZE_MIB} MiB upload limit"
        )

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staging_path = None
    published_path = None
    committed = False
    try:
        # Stage on the same volume as public files, but outside the static mount.
        with NamedTemporaryFile(dir=STAGING_DIR, suffix=".mp4", delete=False) as target:
            staging_path = Path(target.name)
            total = 0
            while chunk := file.file.read(CHUNK_SIZE):
                total += len(chunk)
                if total > MAX_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Trailer exceeds the {MAX_SIZE_MIB} MiB upload limit",
                    )
                target.write(chunk)

        if total == 0:
            raise HTTPException(status_code=422, detail="Trailer file is empty")
        validate_mp4(staging_path)
        filename = f"movie_{movie_id}_{uuid4().hex}.mp4"
        destination = PUBLIC_DIR / filename
        os.replace(staging_path, destination)
        published_path = destination
        yield f"/trailers/{filename}"
        committed = True
    finally:
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)
        if published_path is not None and not committed:
            published_path.unlink(missing_ok=True)
