from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

VIDEO_ROOT = Path(os.getenv("VIDEO_ROOT", "/videos")).resolve()
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "/output")).resolve()
VIDEO_JSON_PREFIX = os.getenv("VIDEO_JSON_PREFIX", "../")
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg"}

app = FastAPI(title="Volley Court Calibrator", version="1.0.0")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

COURT_NAMES = [
    "near_left",
    "near_right",
    "near_attack_left",
    "near_attack_right",
    "centre_left",
    "centre_right",
    "far_attack_left",
    "far_attack_right",
    "far_left",
    "far_right",
]

# Standard indoor volleyball court: 18 m long x 9 m wide.
# The attack lines are 3 m from the centre line, i.e. 6 m from each baseline.
COURT_WORLD = {
    "near_left": (0.0, 0.0),
    "near_right": (9.0, 0.0),
    "near_attack_left": (0.0, 6.0),
    "near_attack_right": (9.0, 6.0),
    "centre_left": (0.0, 9.0),
    "centre_right": (9.0, 9.0),
    "far_attack_left": (0.0, 12.0),
    "far_attack_right": (9.0, 12.0),
    "far_left": (0.0, 18.0),
    "far_right": (9.0, 18.0),
}


class CalibrationRequest(BaseModel):
    video_id: str
    kind: str
    frame_size: list[int] = Field(min_length=2, max_length=2)
    reference_frame: int = Field(ge=0)
    landmarks: dict[str, list[float]]


def encode_video_id(relative_path: str) -> str:
    return base64.urlsafe_b64encode(relative_path.encode("utf-8")).decode("ascii").rstrip("=")


def decode_video_id(video_id: str) -> str:
    try:
        padding = "=" * (-len(video_id) % 4)
        relative_path = base64.urlsafe_b64decode(video_id + padding).decode("utf-8")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid video identifier") from exc

    # IDs are relative POSIX paths. Reject absolute paths and traversal.
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise HTTPException(status_code=400, detail="Invalid video path")
    return pure.as_posix()


def get_video_path(video_id: str) -> tuple[Path, str]:
    relative_path = decode_video_id(video_id)
    path = (VIDEO_ROOT / Path(*PurePosixPath(relative_path).parts)).resolve()
    try:
        path.relative_to(VIDEO_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Video path is outside the video directory") from exc
    if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Video not found")
    return path, relative_path


def get_output_dir(relative_path: str) -> Path:
    source = PurePosixPath(relative_path)
    parent = Path(*source.parent.parts) if source.parent != PurePosixPath(".") else Path()
    folder = (OUTPUT_ROOT / parent / source.stem).resolve()
    try:
        folder.relative_to(OUTPUT_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid output path") from exc
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def json_video_path(relative_path: str) -> str:
    # Mirrors the relative path in the mounted videos folder and defaults to
    # the ../ prefix used by the user's existing calibration JSON example.
    return f"{VIDEO_JSON_PREFIX}{relative_path}"


def validate_landmarks(landmarks: dict[str, list[float]], required: list[str], width: int, height: int) -> None:
    missing = [name for name in required if name not in landmarks]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing landmarks: {', '.join(missing)}")
    for name in required:
        point = landmarks[name]
        if not isinstance(point, list) or len(point) != 2:
            raise HTTPException(status_code=422, detail=f"Landmark '{name}' must contain [x, y]")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Landmark '{name}' has invalid coordinates") from exc
        if not np.isfinite([x, y]).all():
            raise HTTPException(status_code=422, detail=f"Landmark '{name}' has non-finite coordinates")
        if x < 0 or y < 0 or x >= width or y >= height:
            raise HTTPException(status_code=422, detail=f"Landmark '{name}' is outside the video frame")


def calculate_court_rmse(landmarks: dict[str, list[float]]) -> float:
    image_points = np.asarray([landmarks[name] for name in COURT_NAMES], dtype=np.float64)
    court_points = np.asarray([COURT_WORLD[name] for name in COURT_NAMES], dtype=np.float64)
    homography, _ = cv2.findHomography(image_points, court_points, method=0)
    if homography is None:
        raise HTTPException(status_code=422, detail="Could not calculate court homography. Check the landmark positions.")
    projected = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - court_points, axis=1)
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    if not np.isfinite(rmse):
        raise HTTPException(status_code=422, detail="Court calibration produced an invalid RMSE")
    return round(rmse, 4)


@app.get("/", response_class=HTMLResponse)
def home() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/videos")
def list_videos() -> dict[str, Any]:
    if not VIDEO_ROOT.exists():
        return {"videos": [], "video_root": str(VIDEO_ROOT), "warning": "Video directory does not exist inside the container."}

    videos = []
    for path in VIDEO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        try:
            relative_path = path.relative_to(VIDEO_ROOT).as_posix()
            stat = path.stat()
        except (OSError, ValueError):
            continue
        videos.append({
            "id": encode_video_id(relative_path),
            "name": path.name,
            "relative_path": relative_path,
            "size_bytes": stat.st_size,
        })
    videos.sort(key=lambda item: item["relative_path"].lower())
    return {"videos": videos, "video_root": str(VIDEO_ROOT)}


@app.get("/api/videos/{video_id}/metadata")
def video_metadata(video_id: str) -> dict[str, Any]:
    path, relative_path = get_video_path(video_id)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise HTTPException(status_code=422, detail="OpenCV could not read this video")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=422, detail="Could not determine video dimensions")
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    return {
        "relative_path": relative_path,
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "frame_count": frame_count,
        "duration_seconds": round(frame_count / fps, 3) if frame_count > 0 else None,
    }


def file_iterator(path: Path, start: int, length: int, chunk_size: int = 1024 * 1024):
    with path.open("rb") as file:
        file.seek(start)
        remaining = length
        while remaining > 0:
            chunk = file.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@app.get("/api/videos/{video_id}/stream")
async def stream_video(video_id: str, request: Request):
    path, _ = get_video_path(video_id)
    file_size = path.stat().st_size
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    range_header = request.headers.get("range")

    if not range_header:
        return StreamingResponse(
            file_iterator(path, 0, file_size),
            media_type=content_type,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )

    if not range_header.startswith("bytes=") or "," in range_header:
        return range_not_satisfiable(file_size)
    try:
        range_value = range_header.removeprefix("bytes=").strip()
        start_text, end_text = range_value.split("-", 1)
        if start_text == "":
            suffix_length = int(end_text)
            if suffix_length <= 0:
                return range_not_satisfiable(file_size)
            start = max(file_size - suffix_length, 0)
            end = file_size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
        if start < 0 or start >= file_size or end < start:
            return range_not_satisfiable(file_size)
        end = min(end, file_size - 1)
    except (ValueError, TypeError):
        return range_not_satisfiable(file_size)

    length = end - start + 1
    return StreamingResponse(
        file_iterator(path, start, length),
        status_code=206,
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(length),
        },
    )


def range_not_satisfiable(file_size: int) -> Response:
    return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}", "Accept-Ranges": "bytes"})


@app.post("/api/calibration")
def save_calibration(payload: CalibrationRequest) -> JSONResponse:
    if payload.kind not in {"court", "net"}:
        raise HTTPException(status_code=400, detail="kind must be 'court' or 'net'")
    path, relative_path = get_video_path(payload.video_id)
    width, height = int(payload.frame_size[0]), int(payload.frame_size[1])
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=422, detail="Invalid frame size")

    required = COURT_NAMES if payload.kind == "court" else ["net_left", "net_right"]
    validate_landmarks(payload.landmarks, required, width, height)

    # Keep only the schema's named landmarks and preserve the input coordinate precision.
    clean_landmarks = {name: [float(payload.landmarks[name][0]), float(payload.landmarks[name][1])] for name in required}
    document: dict[str, Any] = {
        "video": json_video_path(relative_path),
        "frame_size": [width, height],
        "landmarks": clean_landmarks,
        "extra": [],
    }
    if payload.kind == "court":
        document["rmse_m"] = calculate_court_rmse(clean_landmarks)

    output_dir = get_output_dir(relative_path)
    output_file = output_dir / f"{payload.kind}.json"
    try:
        output_file.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not write output file: {exc}") from exc

    return JSONResponse({
        "ok": True,
        "kind": payload.kind,
        "file": f"{payload.kind}.json",
        "output_path": output_file.relative_to(OUTPUT_ROOT).as_posix(),
        "document": document,
    })
