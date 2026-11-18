"""Turn labels into training data.

  dataset_<ts>/ball_yolo/        YOLO detection dataset (class 0 = ball) -> train your ball detector
  dataset_<ts>/events_cls/       YOLO classification folders (player crops per action) - optional, weak single-frame model
  dataset_<ts>/events.csv        every labelled event (for evaluating / tuning the rule-based event detector)
  dataset_<ts>/labels/           copy of the raw label json files
Train/val are split by TIME BLOCKS of each video (never random frames: neighbouring frames are near-duplicates and would leak).
"""
from __future__ import annotations
import csv, json, shutil, time
from pathlib import Path
import cv2


def ball_samples(ball: dict, max_gap: int, size_override=None):
    """frame -> (x, y, size) for visible, None for explicitly 'not visible'; gaps <= max_gap between visible keyframes are linearly interpolated."""
    keys = sorted(int(k) for k in ball)
    out = {}
    for k in keys:
        b = ball[str(k)]
        out[k] = (b["x"], b["y"], size_override or b.get("s", 16)) if b.get("v", 1) else None
    for a, b in zip(keys[:-1], keys[1:]):
        A, B = ball[str(a)], ball[str(b)]
        if A.get("v", 1) and B.get("v", 1) and 1 < b - a <= max_gap:
            for f in range(a + 1, b):
                t = (f - a) / (b - a)
                s = size_override or (A.get("s", 16) * (1 - t) + B.get("s", 16) * t)
                out[f] = (A["x"] * (1 - t) + B["x"] * t, A["y"] * (1 - t) + B["y"] * t, s)
    return out


def make_split(val_block: int, val_every: int, guard: int):
    def split(f: int) -> str | None:
        blk = f // val_block
        if blk % val_every == val_every - 1:
            return "val"
        nb = (f + guard) // val_block, (f - guard) // val_block
        if any(b % val_every == val_every - 1 for b in nb):
            return None                                   # too close to a val block -> skip to avoid leakage
        return "train"
    return split


def run_export(videos_dir: Path, labels_dir: Path, export_dir: Path, p: dict) -> dict:
    max_gap = int(p.get("max_gap", 3)); size_override = p.get("ball_size") or None
    val_block, val_every, guard = int(p.get("val_block", 90)), int(p.get("val_every", 6)), int(p.get("guard", 10))
    crops = bool(p.get("crops", True)); pad = float(p.get("pad", 0.15))
    split = make_split(val_block, val_every, guard)
    docs = []
    for lp in sorted(Path(labels_dir).glob("*.json")):
        d = json.load(open(lp))
        if (Path(videos_dir) / d["video"]).exists() and (d.get("ball") or d.get("events")):
            if not p.get("videos") or d["video"] in p["videos"]:
                docs.append(d)
    if not docs:
        raise ValueError("no labelled videos found")
    name = "dataset_" + time.strftime("%Y%m%d_%H%M%S")
    root = Path(export_dir) / name
    for sp in ("train", "val"):
        (root / "ball_yolo/images" / sp).mkdir(parents=True); (root / "ball_yolo/labels" / sp).mkdir(parents=True)
    (root / "labels").mkdir()
    stats = dict(ball_train=0, ball_val=0, neg_train=0, neg_val=0, events=0, crops=0, skipped_guard=0)
    rows = []
    for d in docs:
        stem = Path(d["video"]).stem
        shutil.copy(Path(labels_dir) / (d["video"] + ".json"), root / "labels")
        samples = ball_samples(d.get("ball", {}), max_gap, size_override)
        ev = d.get("events", [])
        need = {}
        for f, s in samples.items():
            sp = split(f)
            if sp is None: stats["skipped_guard"] += 1; continue
            need.setdefault(f, {})["ball"] = (sp, s)
        for e in ev:
            sp = split(int(e["frame"]))
            rows.append(dict(video=d["video"], frame=e["frame"], duration=max(1, int(e.get("duration", 5))),
                             time=round(e["frame"] / d["fps"], 3), type=e["type"],
                             x1=(e.get("bbox") or [None] * 4)[0], y1=(e.get("bbox") or [None] * 4)[1],
                             x2=(e.get("bbox") or [None] * 4)[2], y2=(e.get("bbox") or [None] * 4)[3],
                             px=(e.get("point") or [None, None])[0], py=(e.get("point") or [None, None])[1], split=sp or "guard"))
            stats["events"] += 1
            if crops and e.get("bbox") and sp:
                need.setdefault(int(e["frame"]), {}).setdefault("ev", []).append((sp, e))
        cap = cv2.VideoCapture(str(Path(videos_dir) / d["video"])); i = 0
        W, H = int(cap.get(3)), int(cap.get(4))
        for f in sorted(need):
            while i < f:
                cap.grab(); i += 1
            ok, img = cap.read(); i += 1
            if not ok: break
            if "ball" in need[f]:
                sp, s = need[f]["ball"]
                cv2.imwrite(str(root / f"ball_yolo/images/{sp}/{stem}_{f:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                line = ""
                if s is not None:
                    x, y, sz = s
                    x1, y1, x2, y2 = max(0, x - sz / 2), max(0, y - sz / 2), min(W, x + sz / 2), min(H, y + sz / 2)
                    line = f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"
                open(root / f"ball_yolo/labels/{sp}/{stem}_{f:06d}.txt", "w").write(line)
                stats[("ball_" if s else "neg_") + sp] += 1
            for sp, e in need[f].get("ev", []):
                x1, y1, x2, y2 = e["bbox"]; w, h = x2 - x1, y2 - y1
                cx1, cy1, cx2, cy2 = int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h)), int(min(W, x2 + pad * w)), int(min(H, y2 + pad * h))
                if cx2 - cx1 < 8 or cy2 - cy1 < 8: continue
                out = root / "events_cls" / sp / e["type"]; out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / f"{stem}_{f:06d}_{e.get('id', 0)}.jpg"), img[cy1:cy2, cx1:cx2]); stats["crops"] += 1
        cap.release()
    if stats["ball_val"] + stats["neg_val"] == 0:                     # short clips: no val block existed
        imgs = sorted((root / "ball_yolo/images/train").glob("*.jpg"))[::7]
        for im in imgs:
            lb = root / "ball_yolo/labels/train" / (im.stem + ".txt")
            shutil.move(im, root / "ball_yolo/images/val" / im.name); shutil.move(lb, root / "ball_yolo/labels/val" / lb.name)
        stats["warning"] = ("clips too short for time-block validation: every 7th frame moved to val - validation numbers will be "
                            "optimistic. Label more/longer clips for an honest score.")
    with open(root / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "frame", "duration", "time", "type", "x1", "y1", "x2", "y2", "px", "py", "split"])
        w.writeheader(); [w.writerow(r) for r in rows]
    (root / "ball_yolo/data.yaml").write_text(f"path: {(root / 'ball_yolo').resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    zipf = shutil.make_archive(str(Path(export_dir) / name), "zip", root)
    stats.update(dataset=str(root), zip=Path(zipf).name, videos=[d["video"] for d in docs],
                 train_cmd=f"yolo detect train data={(root / 'ball_yolo/data.yaml').resolve()} model=yolo11s.pt imgsz=1280 epochs=80 batch=4 "
                           "flipud=0 degrees=0 shear=0 perspective=0 mosaic=0.5 hsv_v=0.4")
    return stats
