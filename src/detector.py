import json
import os
import shutil
import subprocess
import sys
import uuid

import cv2
import numpy as np
from ultralytics import YOLO

import storage
from counter import LineCounter
from rules import (DwellRule, RedLightRule, SignalReader, SpeedRule,
                   WrongWayRule)

VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# every position is a fraction of the frame so one config fits any resolution
DEFAULTS = {
    "conf": 0.4,
    "rules": ["red_light", "wrong_way", "speeding", "illegal_stop"],
    "stop_line": 0.6,
    "signal_roi": [0.45, 0.05, 0.5, 0.15],
    "allowed_direction": "down",
    "min_wrong_way_travel": 0.08,
    "speed_limit_kmh": 40,
    "metres_per_pixel": 0.05,
    "no_stop_zone": [[0.05, 0.65], [0.35, 0.65], [0.35, 0.95], [0.05, 0.95]],
    "max_dwell_seconds": 8,
}

GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 215, 255)


def load_config(path=None):
    cfg = dict(DEFAULTS)
    if path and os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def to_pixels(cfg, width, height):
    x1, y1, x2, y2 = cfg["signal_roi"]
    return {
        "stop_y": int(height * cfg["stop_line"]),
        "roi": [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)],
        "zone": [[int(px * width), int(py * height)] for px, py in cfg["no_stop_zone"]],
        "min_travel": int(height * cfg["min_wrong_way_travel"]),
    }


def draw_setup(frame, cfg):
    """Draws the line, signal box and zone so the user can check their setup."""
    h, w = frame.shape[:2]
    px = to_pixels(cfg, w, h)
    out = frame.copy()
    cv2.line(out, (0, px["stop_y"]), (w, px["stop_y"]), RED, 2)
    x1, y1, x2, y2 = px["roi"]
    cv2.rectangle(out, (x1, y1), (x2, y2), YELLOW, 2)
    cv2.polylines(out, [np.array(px["zone"], dtype=np.int32)],
                  True, (255, 128, 0), 2)
    return out


def make_web_friendly(path):
    if shutil.which("ffmpeg") is None:
        return path
    fixed = path.replace(".mp4", "_web.mp4")
    done = subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", fixed],
        capture_output=True,
    )
    return fixed if done.returncode == 0 else path


def save_evidence(frame, box, rule, tid, frame_no, folder):
    shot = frame.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(shot, (x1, y1), (x2, y2), RED, 3)
    cv2.putText(shot, rule.replace("_", " "), (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
    path = os.path.join(folder, f"{rule}_{tid}_{frame_no}.jpg")
    cv2.imwrite(path, shot)
    return path


def process_video(source, cfg=None, out_dir="runs", on_progress=None):
    cfg = cfg or load_config()
    run_id = uuid.uuid4().hex[:8]
    run_dir = os.path.join(out_dir, run_id)
    evidence_dir = os.path.join(run_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"could not open {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    px = to_pixels(cfg, width, height)

    enabled = set(cfg["rules"])
    reader = SignalReader(px["roi"]) if "red_light" in enabled else None
    red_light = RedLightRule(px["stop_y"], cfg["allowed_direction"])
    wrong_way = WrongWayRule(cfg["allowed_direction"], px["min_travel"])
    speed = SpeedRule(fps, cfg["metres_per_pixel"], cfg["speed_limit_kmh"])
    dwell = DwellRule(px["zone"], fps, cfg["max_dwell_seconds"])
    volume = LineCounter(px["stop_y"])

    out_path = os.path.join(run_dir, "result.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    db = storage.connect(os.path.join(out_dir, "violations.db"))
    model = YOLO("yolov8n.pt")

    offenders = set()
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        clean = frame.copy()
        signal = reader.state(frame) if reader else "off"

        result = model.track(frame, persist=True, classes=list(VEHICLES),
                             conf=cfg["conf"], verbose=False)[0]

        if result.boxes.id is not None:
            for box, tid, cls in zip(result.boxes.xyxy, result.boxes.id, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                tid = int(tid)
                vehicle = VEHICLES[int(cls)]
                # bottom centre is where the tyres meet the road
                x, y = (x1 + x2) // 2, y2
                volume.update(tid, vehicle, y)

                hits = []
                if "red_light" in enabled and red_light.check(tid, x, y, signal):
                    hits.append(("red_light", "crossed the stop line on red"))
                if "wrong_way" in enabled and wrong_way.check(tid, x, y):
                    hits.append(("wrong_way", "moving against traffic flow"))
                if "speeding" in enabled and speed.check(tid, x, y, frame_no):
                    hits.append(("speeding", f"{speed.speeds[tid]:.0f} km/h"))
                if "illegal_stop" in enabled and dwell.check(tid, x, y, frame_no):
                    hits.append(("illegal_stop", "stationary in no stopping zone"))

                for rule, detail in hits:
                    offenders.add(tid)
                    shot = save_evidence(clean, (x1, y1, x2, y2), rule, tid,
                                         frame_no, evidence_dir)
                    storage.add(db, run_id, rule, tid, vehicle,
                                frame_no / fps, detail, shot)

                colour = RED if tid in offenders else GREEN
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"{vehicle} #{tid}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        frame = draw_setup(frame, cfg)
        shown = {"red": RED, "green": GREEN, "yellow": YELLOW}.get(signal, (255, 255, 255))
        cv2.putText(frame, f"signal: {signal}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, shown, 2)
        cv2.putText(frame, f"violations: {len(offenders)}", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writer.write(frame)
        if on_progress:
            on_progress(min(frame_no / total, 1.0))

    cap.release()
    writer.release()
    video = make_web_friendly(out_path)
    return run_id, video, storage.for_run(db, run_id), db


if __name__ == "__main__":
    src = sys.argv[1]
    cfg = load_config(sys.argv[2] if len(sys.argv) > 2 else None)
    run_id, video, rows, _ = process_video(src, cfg)
    print("run:", run_id)
    print("video:", video)
    for r in rows:
        print(f"{r['video_second']:>7.2f}s  {r['rule']:<13} #{r['track_id']} {r['vehicle']}  {r['detail']}")
    print(len(rows), "violations")
