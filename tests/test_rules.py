import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rules import (DwellRule, FlowField, GroundPlane, RedLightRule, SignalReader,
                   SpeedRule, WrongWayRule)


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


def test_red_light_ignores_opposite_direction():
    rule = RedLightRule(stop_line_y=100, direction="down")
    rule.check(1, 0, 120, "red")
    assert rule.check(1, 0, 90, "red") is False


def test_ground_plane_maps_to_metres():
    plane = GroundPlane([(100, 100), (300, 100), (300, 300), (100, 300)], 3.5, 10)
    x, y = plane.to_metres(200, 200)
    assert abs(x - 1.75) < 0.01
    assert abs(y - 5.0) < 0.01


def test_speed_flags_fast_vehicle_only():
    # 1000 pixels cover 100 metres, so one pixel is 0.1 m
    plane = GroundPlane([(0, 0), (1000, 0), (1000, 1000), (0, 1000)], 100, 100)
    slow = SpeedRule(10, plane, limit_kmh=40)
    fast = SpeedRule(10, plane, limit_kmh=40)
    slow_hit = fast_hit = False
    for i in range(30):
        slow_hit |= slow.check(1, 500, i * 3, i)
        fast_hit |= fast.check(2, 500, i * 20, i)
    assert slow_hit is False
    assert fast_hit is True


def test_speed_ignores_tracker_jump():
    plane = GroundPlane([(0, 0), (1000, 0), (1000, 1000), (0, 1000)], 100, 100)
    rule = SpeedRule(10, plane, limit_kmh=40)
    hit = False
    for i in range(20):
        hit |= rule.check(1, 100, 100 + i, i)
    hit |= rule.check(1, 100, 900, 20)
    assert hit is False


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


def _drive(rule, tid, x, y_start, step, frames):
    hits = False
    for i in range(frames):
        hits |= rule.check(tid, x, y_start + i * step)
    return hits


def test_two_way_road_flags_nobody_in_normal_traffic():
    # right hand traffic: left lane goes up the frame, right lane goes down
    flow = FlowField(200, 200, min_samples=20)
    rule = WrongWayRule(flow, fps=10, min_motion=5, min_travel=10)
    flagged = False
    for car in range(12):
        flagged |= _drive(rule, car, 50, 190, -4, 45)
        flagged |= _drive(rule, 100 + car, 150, 10, 4, 45)
    assert flagged is False


def test_wrong_way_driver_is_caught_in_either_lane():
    flow = FlowField(200, 200, min_samples=20)
    rule = WrongWayRule(flow, fps=10, min_motion=5, min_travel=10)
    for car in range(12):
        _drive(rule, car, 50, 190, -4, 45)
        _drive(rule, 100 + car, 150, 10, 4, 45)

    assert _drive(rule, 900, 150, 190, -4, 40) is True
    assert _drive(rule, 901, 50, 10, 4, 40) is True


def test_wrong_way_works_on_left_hand_traffic_too():
    flow = FlowField(200, 200, min_samples=20)
    rule = WrongWayRule(flow, fps=10, min_motion=5, min_travel=10)
    for car in range(12):
        _drive(rule, car, 50, 10, 4, 45)
        _drive(rule, 100 + car, 150, 190, -4, 45)
    assert _drive(rule, 77, 50, 190, -4, 40) is True
    assert _drive(rule, 78, 150, 10, 4, 40) is True


def test_unlearned_cells_never_flag():
    flow = FlowField(200, 200, min_samples=500)
    rule = WrongWayRule(flow, fps=10, min_motion=5, min_travel=10)
    assert _drive(rule, 1, 50, 190, -4, 40) is False
