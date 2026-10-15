import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rules import DwellRule, RedLightRule, SignalReader, SpeedRule, WrongWayRule


def test_red_light_only_counts_on_red():
    rule = RedLightRule(stop_line_y=100)
    rule.check(1, 50, 90, "green")
    assert rule.check(1, 50, 110, "green") is False

    rule.check(2, 50, 90, "red")
    assert rule.check(2, 50, 110, "red") is True


def test_red_light_flags_vehicle_once():
    rule = RedLightRule(stop_line_y=100)
    rule.check(1, 0, 90, "red")
    assert rule.check(1, 0, 110, "red") is True
    rule.check(1, 0, 90, "red")
    assert rule.check(1, 0, 110, "red") is False


def test_wrong_way_needs_enough_travel():
    rule = WrongWayRule("down", min_travel=50)
    rule.check(1, 0, 300)
    assert rule.check(1, 0, 280) is False
    assert rule.check(1, 0, 230) is True


def test_wrong_way_ignores_correct_direction():
    rule = WrongWayRule("down", min_travel=50)
    rule.check(1, 0, 100)
    assert rule.check(1, 0, 400) is False


def test_speed_detects_fast_vehicle():
    fps = 10
    rule = SpeedRule(fps, metres_per_pixel=0.1, limit_kmh=40)
    flagged = False
    for i in range(30):
        # 3 pixels per frame at 10 fps and 0.1 m per pixel is 3 m/s, so it stays under the limit
        flagged |= rule.check(1, 0, i * 3, i)
    assert flagged is False

    fast = SpeedRule(fps, metres_per_pixel=0.1, limit_kmh=40)
    for i in range(30):
        # 20 pixels per frame is 20 m/s, about 72 km/h
        flagged |= fast.check(2, 0, i * 20, i)
    assert flagged is True


def test_dwell_flags_parked_vehicle():
    zone = [[0, 0], [100, 0], [100, 100], [0, 100]]
    rule = DwellRule(zone, fps=10, max_seconds=2)
    flagged = False
    for frame in range(40):
        flagged |= rule.check(1, 50, 50, frame)
    assert flagged is True


def test_dwell_resets_when_vehicle_leaves():
    zone = [[0, 0], [100, 0], [100, 100], [0, 100]]
    rule = DwellRule(zone, fps=10, max_seconds=2)
    for frame in range(15):
        rule.check(1, 50, 50, frame)
    rule.check(1, 500, 500, 16)
    assert rule.check(1, 50, 50, 17) is False


def test_signal_reader_sees_red_and_green():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    reader = SignalReader([10, 10, 60, 60])

    frame[10:60, 10:60] = (0, 0, 255)
    assert reader.state(frame) == "red"

    frame[10:60, 10:60] = (0, 255, 0)
    assert reader.state(frame) == "green"

    frame[10:60, 10:60] = (0, 0, 0)
    assert reader.state(frame) == "unknown"
