from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, Response

from portal.data_manager import create_data_manager_app

ROOT = Path(__file__).resolve().parents[1]
VOLLEY_LABELER_ROOT = ROOT / "volley-labeler"
PORTAL_ROOT = Path(__file__).resolve().parent


def _load_package(name: str, package_dir: Path):
    package = types.ModuleType(name)
    package.__path__ = [str(package_dir)]
    package.__package__ = name
    sys.modules[name] = package
    return importlib.import_module(f"{name}.main")


def _load_volley_labeler():
    sys.path.insert(0, str(VOLLEY_LABELER_ROOT))
    spec = importlib.util.spec_from_file_location("volley_labeler", VOLLEY_LABELER_ROOT / "app.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Volley Labeler")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.cfg.update(
        videos=Path(os.getenv("LABELER_VIDEO_ROOT", "/videos")),
        labels=Path(os.getenv("LABELER_LABELS_ROOT", "/labels")),
        export=Path(os.getenv("LABELER_EXPORT_ROOT", "/exports")),
        weights=os.getenv("LABELER_WEIGHTS") or None,
    )
    return module


volley_labeler = _load_volley_labeler()
calibrator = _load_package("volley_calibrator", ROOT / "volley-calibrator" / "app")
training = _load_package("volley_training", ROOT / "volley-training-studio" / "app")
analyzer = _load_package("volley_analyzer", ROOT / "volley-analyzer" / "app")

data_manager = create_data_manager_app(
    videos_dir=volley_labeler.cfg["videos"],
    labels_dir=volley_labeler.cfg["labels"],
    model_roots=analyzer.MODEL_ROOTS,
    video_cache=volley_labeler._videos,
    cache_lock=volley_labeler._lock,
)

app = FastAPI(title="VolleyVision", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def home() -> FileResponse:
    return FileResponse(PORTAL_ROOT / "static" / "index.html")


@app.get("/_portal/path-prefix.js")
def path_prefix_script() -> Response:
    script = """(() => {
  const prefix = window.location.pathname.replace(/\\/+$/, "");
  const localApi = value => typeof value === "string" && value.startsWith("/api/") ? prefix + value : value;
  const nativeFetch = window.fetch.bind(window);
  window.fetch = (input, init) => nativeFetch(localApi(input), init);
  const mediaSrc = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, "src");
  Object.defineProperty(HTMLMediaElement.prototype, "src", {
    configurable: true,
    enumerable: mediaSrc.enumerable,
    get() { return mediaSrc.get.call(this); },
    set(value) { mediaSrc.set.call(this, localApi(value)); }
  });
    const rewriteMedia = root => {
        const sources = root instanceof HTMLMediaElement ? [root] : root.querySelectorAll?.("audio[src], video[src]") || [];
        for (const media of sources) {
            const source = media.getAttribute("src");
            const rewritten = localApi(source);
            if (rewritten !== source) media.setAttribute("src", rewritten);
        }
    };
    new MutationObserver(records => records.forEach(record => record.addedNodes.forEach(rewriteMedia)))
        .observe(document.documentElement, {childList: true, subtree: true});
  document.addEventListener("click", event => {
    if (!(event.target instanceof Element)) return;
    const link = event.target.closest('a[href^="/api/"]');
    if (link) link.href = localApi(link.getAttribute("href"));
  }, true);
})();"""
    return Response(script, media_type="application/javascript")


@app.get("/_portal/shared.css")
def shared_stylesheet() -> FileResponse:
    return FileResponse(PORTAL_ROOT / "static" / "shared.css", media_type="text/css")


@app.get("/_portal/volleyvision-logo.svg")
def logo() -> FileResponse:
    return FileResponse(
        PORTAL_ROOT / "static" / "volleyvision-logo.svg",
        media_type="image/svg+xml",
    )


app.mount("/labeler", volley_labeler.app, name="volley_labeler")
app.mount("/court", calibrator.app, name="court")
app.mount("/training", training.app, name="training")
app.mount("/analysis", analyzer.app, name="analysis")
app.mount("/data", data_manager, name="data")