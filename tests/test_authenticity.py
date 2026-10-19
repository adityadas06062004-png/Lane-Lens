import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from authenticity import check_source

W, H, FRAMES = 320, 240, 60


def _background(seed):
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, (H + 40, W + 400, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (9, 9), 0)


def _write(path, make_frame):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (W, H))
    for i in range(FRAMES):
        writer.write(make_frame(i))
    writer.release()


def _status(result, name):
    return next(c["status"] for c in result["checks"] if c["name"] == name)


def test_fixed_camera_with_moving_car_passes(tmp_path):
    bg = _background(1)

    def frame(i):
        img = bg[:H, :W].copy()
        cv2.rectangle(img, (40 + i * 3, 100), (80 + i * 3, 130), (0, 0, 255), -1)
        return img

    path = str(tmp_path / "fixed.avi")
    _write(path, frame)
    result = check_source(path)
    assert _status(result, "Fixed camera") == "ok"
    assert _status(result, "Continuous footage") == "ok"
    assert result["verdict"] != "block"
    assert len(result["sha256"]) == 64


def test_panning_camera_is_rejected(tmp_path):
    bg = _background(2)

    def frame(i):
        return bg[:H, i * 5:i * 5 + W].copy()

    path = str(tmp_path / "pan.avi")
    _write(path, frame)
    result = check_source(path)
    assert _status(result, "Fixed camera") == "fail"
    assert result["verdict"] == "block"


def test_scene_cut_is_rejected(tmp_path):
    first, second = _background(3), np.full((H + 40, W + 400, 3), 230, np.uint8)

    def frame(i):
        return (first if i < FRAMES // 2 else second)[:H, :W].copy()

    path = str(tmp_path / "cut.avi")
    _write(path, frame)
    result = check_source(path)
    assert _status(result, "Continuous footage") == "fail"
    assert result["verdict"] == "block"


def test_unreadable_file_is_rejected(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    assert check_source(str(path))["verdict"] == "block"
