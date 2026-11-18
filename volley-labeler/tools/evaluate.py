"""Score volleyscout against YOUR labels - this is how you know if a change (new weights, new thresholds) really helped.

python tools/evaluate.py --labels labels/clip.mp4.json --analysis out/analysis.json --ball out/ball.json
Label EVERY event in the stretch you evaluate on (otherwise correct detections count as false positives).
Same video file must be used for labelling and analysis (frame numbers must match).
"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True); ap.add_argument("--analysis"); ap.add_argument("--ball")
ap.add_argument("--tol", type=float, default=0.3, help="seconds: max time difference to count an event as found")
ap.add_argument("--px", type=float, default=12, help="pixels: max distance to count a ball position as correct")
ap.add_argument("--start", type=float); ap.add_argument("--end", type=float)
a = ap.parse_args()
doc = json.load(open(a.labels)); fps = doc["fps"]

if a.analysis:
    pred_all = json.load(open(a.analysis))["events"]
    gt = [e for e in doc["events"] if e["type"] != "bounce"]
    if not gt:
        print("no labelled events"); 
    else:
        t0 = a.start if a.start is not None else min(e["frame"] for e in gt) / fps - 1.0
        t1 = a.end if a.end is not None else max(e["frame"] for e in gt) / fps + 1.0
        pred = [p for p in pred_all if t0 <= p["frame"] / fps <= t1 and p["type"] != "bounce"]
        gt = [e for e in gt if t0 <= e["frame"] / fps <= t1]
        used = set(); tp = {}; fn = {}; fp = {}; conf = {}
        for g in sorted(gt, key=lambda e: e["frame"]):
            c = [(abs(p["frame"] - g["frame"]), i) for i, p in enumerate(pred) if i not in used and p["type"] == g["type"] and abs(p["frame"] - g["frame"]) / fps <= a.tol]
            if c:
                used.add(min(c)[1]); tp[g["type"]] = tp.get(g["type"], 0) + 1
            else:
                fn[g["type"]] = fn.get(g["type"], 0) + 1
                wrong = [p["type"] for p in pred if abs(p["frame"] - g["frame"]) / fps <= a.tol]
                conf[(g["type"], wrong[0] if wrong else "(nothing)")] = conf.get((g["type"], wrong[0] if wrong else "(nothing)"), 0) + 1
        for i, p in enumerate(pred):
            if i not in used: fp[p["type"]] = fp.get(p["type"], 0) + 1
        print(f"\nEVENTS  window {t0:.1f}s-{t1:.1f}s, tolerance ±{a.tol}s   ({len(gt)} labelled, {len(pred)} predicted)")
        print(f"{'class':<10}{'TP':>5}{'FN':>5}{'FP':>5}{'precision':>11}{'recall':>9}{'F1':>7}")
        T = [0, 0, 0]
        for c in sorted(set(tp) | set(fn) | set(fp)):
            t, n, f = tp.get(c, 0), fn.get(c, 0), fp.get(c, 0); T[0] += t; T[1] += n; T[2] += f
            pr, rc = t / max(1, t + f), t / max(1, t + n)
            print(f"{c:<10}{t:>5}{n:>5}{f:>5}{pr:>11.2f}{rc:>9.2f}{2 * pr * rc / max(1e-9, pr + rc):>7.2f}")
        pr, rc = T[0] / max(1, T[0] + T[2]), T[0] / max(1, T[0] + T[1])
        print(f"{'ALL':<10}{T[0]:>5}{T[1]:>5}{T[2]:>5}{pr:>11.2f}{rc:>9.2f}{2 * pr * rc / max(1e-9, pr + rc):>7.2f}")
        if conf:
            print("\nmissed labelled events -> what was predicted at that moment:")
            for (g, p), n in sorted(conf.items(), key=lambda kv: -kv[1]): print(f"  {g:<10} -> {p:<10} x{n}")

if a.ball:
    B = {int(k): v for k, v in json.load(open(a.ball)).items()}
    vis = {int(k): v for k, v in doc["ball"].items() if v.get("v", 1)}
    neg = [int(k) for k, v in doc["ball"].items() if not v.get("v", 1)]
    err, found, interp = [], 0, 0
    for f, l in vis.items():
        if f in B:
            e = float(np.hypot(B[f][0] - l["x"], B[f][1] - l["y"])); err.append(e)
            if e <= a.px: found += 1; interp += B[f][2] == 2
    print(f"\nBALL  {len(vis)} labelled frames, correct within {a.px:.0f}px")
    print(f"  recall (tracked and correct):   {found / max(1, len(vis)):.2f}   (of which interpolated: {interp})")
    print(f"  tracked but wrong (> {a.px:.0f}px):      {sum(e > a.px for e in err) / max(1, len(vis)):.2f}")
    print(f"  not tracked at all:             {sum(f not in B for f in vis) / max(1, len(vis)):.2f}")
    if err: print(f"  median error: {np.median(err):.1f}px")
    if neg: print(f"  'not visible' frames where a ball was still predicted (false positives): {sum(f in B for f in neg)}/{len(neg)}")
