import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import detector
import storage

W, H, FRAMES = 320, 240, 260


class FakeModel:
    """Stands in for YOLO. Cars drive in two lanes and one drives the wrong way late on."""

    def __init__(self):
        self.frame = 0

    def track(self, frame, **kwargs):
        self.frame += 1
        f = self.frame
        ids, boxes = [], []

        def car(tid, cx, y):
            ids.append(tid)
            boxes.append([cx - 15, y - 25, cx + 15, y])

        for n in range(12):
            start = n * 12
            if 0 <= f - start < 50:
                car(n + 1, 80, 230 - (f - start) * 4)
                car(100 + n + 1, 240, 30 + (f - start) * 4)
        if f > 200:
            car(999, 240, 230 - (f - 200) * 4)

        boxes_obj = SimpleNamespace(id=ids or None, xyxy=boxes, cls=[2] * len(ids))
        return [SimpleNamespace(boxes=boxes_obj)]


def _make_clip(path, pan=False):
    rng = np.random.default_rng(5)
    bg = cv2.GaussianBlur(rng.integers(40, 200, (H, W + 1600, 3), dtype=np.uint8), (9, 9), 0)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (W, H))
    for i in range(FRAMES):
        shift = i * 5 if pan else 0
        writer.write(bg[:, shift:shift + W].copy())
    writer.release()


def _cfg():
    cfg = detector.load_config()
    cfg["rules"] = ["wrong_way"]
    cfg["flow_min_samples"] = 20
    cfg["device"] = "cpu"
    return cfg


def test_two_way_traffic_only_flags_the_wrong_way_car(tmp_path, monkeypatch):
    monkeypatch.setattr(detector, "load_model", lambda name: FakeModel())
    clip = str(tmp_path / "road.avi")
    _make_clip(clip)

    out = detector.process_video(clip, _cfg(), out_dir=str(tmp_path / "runs"))

    assert out["authenticity"]["verdict"] != "block"
    flagged = {v["track_id"] for v in out["violations"]}
    assert flagged == {999}
    assert os.path.exists(out["violations"][0]["evidence"])

    db = storage.connect(str(tmp_path / "runs" / "violations.db"))
    run = storage.get_run(db, out["run_id"])
    assert len(run["sha256"]) == 64


def test_rejected_clip_is_not_analysed(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(detector, "load_model", lambda name: called.append(1) or FakeModel())
    clip = str(tmp_path / "pan.avi")
    _make_clip(clip, pan=True)

    out = detector.process_video(clip, _cfg(), out_dir=str(tmp_path / "runs"))

    assert out["authenticity"]["verdict"] == "block"
    assert out["video"] is None
    assert out["violations"] == []
    assert called == []


def test_speeding_is_skipped_without_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(detector, "load_model", lambda name: FakeModel())
    clip = str(tmp_path / "road.avi")
    _make_clip(clip)
    cfg = _cfg()
    cfg["rules"] = ["speeding"]

    out = detector.process_video(clip, cfg, out_dir=str(tmp_path / "runs"))
    assert any("calibrated" in w for w in out["warnings"])
    assert out["violations"] == []
