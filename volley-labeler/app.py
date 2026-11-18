"""Volley Labeler - local web app to label ball positions and volleyball events on your own clips.

    python app.py --videos videos --labels labels            # then open http://127.0.0.1:8000
Labels are stored as one JSON file per video in --labels (plain, diff-able, safe to back up).
"""
from __future__ import annotations
import argparse, json, os, shutil, threading, time
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
HERE = Path(__file__).parent
EVENT_TYPES = ["serve", "reception", "dig", "set", "attack", "block", "bounce"]   # attack = spike

cfg = dict(videos=Path("videos"), labels=Path("labels"), export=Path("export"), weights=None)
app = FastAPI(title="Volley Labeler")
_videos: dict[str, "Video"] = {}
_lock = threading.Lock()
_model = {"m": None, "tried": False}


class Video:
    """Random-access frame server on top of OpenCV with a small JPEG cache and sequential-read fast path."""

    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise HTTPException(500, f"cannot open {path.name}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w, self.h = int(self.cap.get(3)), int(self.cap.get(4))
        self.pos = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.lock = threading.Lock()

    def frame(self, n: int) -> np.ndarray:
        with self.lock:
            if n != self.pos:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, img = self.cap.read()
            self.pos = n + 1 if ok else -1
            if not ok:
                raise HTTPException(404, "frame out of range")
            return img

    def jpg(self, n: int, q: int = 90) -> bytes:
        if n in self.cache:
            self.cache.move_to_end(n)
            return self.cache[n]
        ok, buf = cv2.imencode(".jpg", self.frame(n), [cv2.IMWRITE_JPEG_QUALITY, q])
        b = buf.tobytes()
        self.cache[n] = b
        if len(self.cache) > 400:
            self.cache.popitem(last=False)
        return b


def list_video_files() -> list[Path]:
    return sorted(p for p in cfg["videos"].iterdir() if p.suffix.lower() in VIDEO_EXT) if cfg["videos"].exists() else []


def get_video(name: str) -> Video:
    if name not in {p.name for p in list_video_files()}:        # also blocks path traversal
        raise HTTPException(404, "unknown video")
    with _lock:
        if name not in _videos:
            _videos[name] = Video(cfg["videos"] / name)
        return _videos[name]


def label_path(name: str) -> Path:
    return cfg["labels"] / (name + ".json")


def empty_doc(name: str, v: Video) -> dict:
    return dict(video=name, fps=v.fps, width=v.w, height=v.h, n_frames=v.n, rev=0, ball={}, events=[],
                meta=dict(ball_size=16))


def load_doc(name: str) -> dict:
    p = label_path(name)
    if p.exists():
        return json.load(open(p))
    return empty_doc(name, get_video(name))


# ----------------------------------------------------------------------------- API
@app.get("/api/videos")
def api_videos():
    out = []
    for p in list_video_files():
        lp = label_path(p.name)
        nb = ne = 0
        if lp.exists():
            try:
                d = json.load(open(lp)); nb, ne = len(d.get("ball", {})), len(d.get("events", []))
            except Exception:
                pass
        out.append(dict(name=p.name, ball_labels=nb, events=ne, size_mb=round(p.stat().st_size / 1e6, 1)))
    return dict(videos=out, event_types=EVENT_TYPES, can_suggest=_can_suggest())


@app.get("/api/meta/{name}")
def api_meta(name: str):
    v = get_video(name)
    return dict(name=name, fps=v.fps, n=v.n, w=v.w, h=v.h)


@app.get("/api/frame/{name}/{n}.jpg")
def api_frame(name: str, n: int):
    v = get_video(name)
    if n < 0 or (v.n and n >= v.n):
        raise HTTPException(404, "frame out of range")
    return Response(v.jpg(n), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/labels/{name}")
def api_get_labels(name: str):
    return load_doc(name)


@app.put("/api/labels/{name}")
async def api_put_labels(name: str, doc: dict):
    get_video(name)
    cfg["labels"].mkdir(parents=True, exist_ok=True)
    doc["video"] = name
    doc["rev"] = int(doc.get("rev", 0)) + 1
    p = label_path(name)
    with _lock:
        if name not in {video.name for video in list_video_files()}:
            raise HTTPException(404, "unknown video")
        tmp = p.with_suffix(".tmp")
        json.dump(doc, open(tmp, "w"))
        if p.exists():
            bak = p.with_suffix(".json.bak")
            if not bak.exists() or time.time() - bak.stat().st_mtime > 600:      # rolling backup every 10 min
                shutil.copy(p, bak)
        os.replace(tmp, p)
    return dict(ok=True, rev=doc["rev"])


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    name = Path(file.filename).name
    if Path(name).suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, "unsupported file type")
    cfg["videos"].mkdir(parents=True, exist_ok=True)
    with open(cfg["videos"] / name, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    _videos.pop(name, None)
    return dict(ok=True, name=name)


def _can_suggest() -> bool:
    if cfg["weights"] is None:
        return False
    try:
        import ultralytics  # noqa
        return True
    except Exception:
        return False


@app.get("/api/suggest/{name}/{n}")
def api_suggest(name: str, n: int, conf: float = 0.05):
    """Model proposals for the current frame (balls + persons) - click one to accept instead of hunting for the ball."""
    if not _can_suggest():
        raise HTTPException(501, "start the server with --weights yolo11s.pt (and ultralytics installed) to enable suggestions")
    if not _model["m"]:
        from ultralytics import YOLO
        _model["m"] = YOLO(cfg["weights"])
    m = _model["m"]
    img = get_video(name).frame(n)
    r = m.predict(img, imgsz=1280, conf=conf, verbose=False)[0]
    balls, persons = [], []
    for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        nm = str(m.names[int(k)]).lower()
        if "ball" in nm:
            balls.append(dict(x=float((x1 + x2) / 2), y=float((y1 + y2) / 2), s=float(max(x2 - x1, y2 - y1)), conf=float(c)))
        elif nm in ("person", "player") and c >= 0.3:
            persons.append(dict(box=[float(x1), float(y1), float(x2), float(y2)], conf=float(c)))
    return dict(balls=balls, persons=persons)


@app.post("/api/export")
def api_export(params: dict):
    from export import run_export
    try:
        res = run_export(cfg["videos"], cfg["labels"], cfg["export"], params)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return res


@app.get("/api/download/{name}")
def api_download(name: str):
    p = cfg["export"] / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, filename=p.name)


app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="videos"); ap.add_argument("--labels", default="labels"); ap.add_argument("--export", default="export")
    ap.add_argument("--weights", default=None, help="optional YOLO weights for click-to-accept suggestions, e.g. yolo11s.pt")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    cfg.update(videos=Path(a.videos), labels=Path(a.labels), export=Path(a.export), weights=a.weights)
    for k in ("videos", "labels", "export"):
        cfg[k].mkdir(parents=True, exist_ok=True)
    print(f"open http://{a.host}:{a.port}   videos={cfg['videos'].resolve()}  labels={cfg['labels'].resolve()}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
