import json
import os
import shutil
import subprocess
import sys
import uuid

import cv2
import numpy as np

import authenticity
import storage
from counter import LineCounter
from rules import (DwellRule, FixedFlow, FlowField, GroundPlane, RedLightRule,
                   SignalReader, SpeedRule, WrongWayRule)

HERE = os.path.dirname(os.path.abspath(__file__))
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# fast traffic uses a bigger model, a larger input size and a tracker that
# tolerates big jumps between frames
PROFILES = {
    "standard": {
        "model": "yolov8n.pt", "imgsz": 640, "conf": 0.4,
        "tracker": None, "speed_window": 1.0,
    },
    "fast_traffic": {
        "model": "yolov8s.pt", "imgsz": 960, "conf": 0.25,
        "tracker": os.path.join(HERE, "fast_bytetrack.yaml"), "speed_window": 0.6,
    },
}

# positions are fractions of the frame so one config fits any resolution
DEFAULTS = {
    "profile": "standard",
    "conf": None,
    "device": "auto",
    "rules": ["red_light", "wrong_way", "illegal_stop"],
    "traffic_flow": "auto",
    "signal_direction": "down",
    "stop_line": 0.6,
    "signal_roi": [0.45, 0.05, 0.5, 0.15],
    "speed_limit_kmh": 40,
    "speed_plane": None,
    "no_stop_zone": [[0.05, 0.65], [0.35, 0.65], [0.35, 0.95], [0.05, 0.95]],
    "max_dwell_seconds": 8,
    "min_motion_frac": 0.008,
    "min_wrong_way_travel": 0.08,
    "flow_min_samples": 40,
    "flow_coherence": 0.6,
    "require_verified_source": True,
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


def load_model(name):
    from ultralytics import YOLO
    return YOLO(name)


def pick_device(cfg):
    if cfg["device"] != "auto":
        return cfg["device"]
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def to_pixels(cfg, width, height):
    x1, y1, x2, y2 = cfg["signal_roi"]
    out = {
        "stop_y": int(height * cfg["stop_line"]),
        "roi": [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)],
        "zone": [[int(px * width), int(py * height)] for px, py in cfg["no_stop_zone"]],
        "min_motion": max(3, int(height * cfg["min_motion_frac"])),
        "min_travel": int(height * cfg["min_wrong_way_travel"]),
        "plane": None,
    }
    plane = cfg.get("speed_plane")
    if plane:
        out["plane"] = [[int(px * width), int(py * height)] for px, py in plane["corners"]]
    return out


def draw_setup(frame, cfg):
    """Draws the line, signal box and zones so the setup can be checked by eye."""
    h, w = frame.shape[:2]
    px = to_pixels(cfg, w, h)
    out = frame.copy()
    cv2.line(out, (0, px["stop_y"]), (w, px["stop_y"]), RED, 2)
    x1, y1, x2, y2 = px["roi"]
    cv2.rectangle(out, (x1, y1), (x2, y2), YELLOW, 2)
    cv2.polylines(out, [np.array(px["zone"], dtype=np.int32)], True, (255, 128, 0), 2)
    if px["plane"]:
        cv2.polylines(out, [np.array(px["plane"], dtype=np.int32)], True, (0, 255, 255), 2)
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


def save_evidence(frame, box, rule, tid, frame_no, folder, unverified):
    shot = frame.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(shot, (x1, y1), (x2, y2), RED, 3)
    cv2.putText(shot, rule.replace("_", " "), (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
    if unverified:
        cv2.putText(shot, "UNVERIFIED SOURCE", (12, shot.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
    path = os.path.join(folder, f"{rule}_{tid}_{frame_no}.jpg")
    cv2.imwrite(path, shot)
    return path


def track_frame(model, frame, profile, conf, state):
    options = dict(persist=True, classes=list(VEHICLES), conf=conf,
                   imgsz=profile["imgsz"], device=state["device"], verbose=False)
    if profile["tracker"]:
        options["tracker"] = profile["tracker"]
    try:
        return model.track(frame, **options)[0]
    except Exception:
        if state["device"] == "cpu":
            raise
        state["device"] = "cpu"
        options["device"] = "cpu"
        return model.track(frame, **options)[0]


def process_video(source, cfg=None, out_dir="runs", on_progress=None, declared_source=""):
    cfg = cfg or load_config()
    profile = PROFILES[cfg["profile"]]
    conf = cfg["conf"] or profile["conf"]
    warnings = []

    run_id = uuid.uuid4().hex[:8]
    run_dir = os.path.join(out_dir, run_id)
    evidence_dir = os.path.join(run_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    db = storage.connect(os.path.join(out_dir, "violations.db"))

    source_check = authenticity.check_source(source)
    storage.add_run(db, run_id, os.path.basename(source), source_check["sha256"],
                    source_check["verdict"], source_check["checks"], declared_source)

    result = {"run_id": run_id, "video": None, "violations": [],
              "authenticity": source_check, "warnings": warnings}

    if source_check["verdict"] == "block" and cfg["require_verified_source"]:
        return result
    unverified = source_check["verdict"] != "pass"

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
    red_light = RedLightRule(px["stop_y"], cfg["signal_direction"])
    dwell = DwellRule(px["zone"], fps, cfg["max_dwell_seconds"])
    volume = LineCounter(px["stop_y"])

    if cfg["traffic_flow"] == "auto":
        flow = FlowField(width, height, min_samples=cfg["flow_min_samples"],
                         coherence=cfg["flow_coherence"])
    else:
        flow = FixedFlow(cfg["traffic_flow"])
    wrong_way = WrongWayRule(flow, fps, px["min_motion"], px["min_travel"])

    speed = None
    if "speeding" in enabled:
        if px["plane"]:
            plane = cfg["speed_plane"]
            ground = GroundPlane(px["plane"], plane["width_m"], plane["length_m"])
            speed = SpeedRule(fps, ground, cfg["speed_limit_kmh"], profile["speed_window"])
        else:
            warnings.append("Speeding was skipped because the road has not been calibrated.")

    out_path = os.path.join(run_dir, "result.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    model = load_model(profile["model"])
    state = {"device": pick_device(cfg)}

    offenders = set()
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        clean = frame.copy()
        signal = reader.state(frame) if reader else "off"

        result_frame = track_frame(model, frame, profile, conf, state)
        boxes = result_frame.boxes

        if boxes.id is not None:
            for box, tid, cls in zip(boxes.xyxy, boxes.id, boxes.cls):
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
                    hits.append(("wrong_way", "moving against the normal flow of its lane"))
                if speed and speed.check(tid, x, y, frame_no):
                    hits.append(("speeding", f"{speed.speeds[tid]:.0f} km/h"))
                if "illegal_stop" in enabled and dwell.check(tid, x, y, frame_no):
                    hits.append(("illegal_stop", "stationary in no stopping zone"))

                for rule, detail in hits:
                    offenders.add(tid)
                    shot = save_evidence(clean, (x1, y1, x2, y2), rule, tid,
                                         frame_no, evidence_dir, unverified)
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
        if unverified:
            cv2.putText(frame, "UNVERIFIED SOURCE", (12, height - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)

        writer.write(frame)
        if on_progress:
            on_progress(min(frame_no / total, 1.0))

    cap.release()
    writer.release()
    result["video"] = make_web_friendly(out_path)
    result["violations"] = storage.for_run(db, run_id)
    return result


if __name__ == "__main__":
    src = sys.argv[1]
    cfg = load_config(sys.argv[2] if len(sys.argv) > 2 else None)
    out = process_video(src, cfg)

    print("run:", out["run_id"])
    print("source check:", out["authenticity"]["verdict"])
    for c in out["authenticity"]["checks"]:
        print(f"  [{c['status']}] {c['name']}: {c['detail']}")
    for w in out["warnings"]:
        print("note:", w)
    if out["video"] is None:
        print("This clip was rejected, so no analysis was run.")
        sys.exit(1)
    print("video:", out["video"])
    for r in out["violations"]:
        print(f"{r['video_second']:>7.2f}s  {r['rule']:<13} #{r['track_id']} {r['vehicle']}  {r['detail']}")
    print(len(out["violations"]), "violations")
