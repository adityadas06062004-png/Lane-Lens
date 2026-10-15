import math
from collections import deque

import cv2
import numpy as np


def inside(point, polygon):
    contour = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


class SignalReader:
    """Guesses the light colour from a rectangle the user drew around the lamp."""

    def __init__(self, roi, min_share=0.03):
        self.roi = roi
        self.min_share = min_share

    def state(self, frame):
        x1, y1, x2, y2 = self.roi
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red = cv2.bitwise_or(cv2.inRange(hsv, (0, 120, 120), (10, 255, 255)),
                             cv2.inRange(hsv, (170, 120, 120), (180, 255, 255)))
        yellow = cv2.inRange(hsv, (15, 120, 120), (35, 255, 255))
        green = cv2.inRange(hsv, (40, 80, 80), (90, 255, 255))

        area = crop.shape[0] * crop.shape[1]
        scores = {
            "red": cv2.countNonZero(red) / area,
            "yellow": cv2.countNonZero(yellow) / area,
            "green": cv2.countNonZero(green) / area,
        }
        best = max(scores, key=scores.get)
        return best if scores[best] >= self.min_share else "unknown"


class RedLightRule:
    name = "red_light"

    def __init__(self, stop_line_y, direction="down"):
        self.stop_line_y = stop_line_y
        self.direction = direction
        self.last_y = {}
        self.flagged = set()

    def check(self, tid, x, y, signal):
        prev = self.last_y.get(tid)
        self.last_y[tid] = y
        if prev is None or tid in self.flagged or signal != "red":
            return False

        if self.direction == "down":
            crossed = prev < self.stop_line_y <= y
        else:
            crossed = prev > self.stop_line_y >= y

        if crossed:
            self.flagged.add(tid)
        return crossed


class WrongWayRule:
    name = "wrong_way"

    def __init__(self, allowed="down", min_travel=80):
        self.allowed = allowed
        self.min_travel = min_travel
        self.first_y = {}
        self.flagged = set()

    def check(self, tid, x, y):
        start = self.first_y.setdefault(tid, y)
        if tid in self.flagged:
            return False

        moved = y - start
        if self.allowed == "down":
            wrong = moved < -self.min_travel
        else:
            wrong = moved > self.min_travel

        if wrong:
            self.flagged.add(tid)
        return wrong


class SpeedRule:
    name = "speeding"

    def __init__(self, fps, metres_per_pixel, limit_kmh, window_seconds=1.0):
        self.fps = fps
        self.mpp = metres_per_pixel
        self.limit = limit_kmh
        self.window = max(int(fps * window_seconds), 2)
        self.history = {}
        self.speeds = {}
        self.flagged = set()

    def check(self, tid, x, y, frame_no):
        trail = self.history.setdefault(tid, deque(maxlen=self.window + 1))
        trail.append((frame_no, x, y))
        if len(trail) <= self.window or tid in self.flagged:
            return False

        f0, x0, y0 = trail[0]
        seconds = (frame_no - f0) / self.fps
        if seconds <= 0:
            return False

        metres = math.hypot(x - x0, y - y0) * self.mpp
        kmh = metres / seconds * 3.6
        self.speeds[tid] = kmh

        if kmh > self.limit:
            self.flagged.add(tid)
            return True
        return False


class DwellRule:
    """Flags a vehicle that sits still inside a no stopping zone for too long."""

    name = "illegal_stop"

    def __init__(self, polygon, fps, max_seconds, still_pixels=15):
        self.polygon = polygon
        self.fps = fps
        self.max_seconds = max_seconds
        self.still_pixels = still_pixels
        self.anchor = {}
        self.flagged = set()

    def check(self, tid, x, y, frame_no):
        if tid in self.flagged:
            return False

        if not inside((x, y), self.polygon):
            self.anchor.pop(tid, None)
            return False

        a = self.anchor.get(tid)
        if a is None or math.hypot(x - a[1], y - a[2]) > self.still_pixels:
            self.anchor[tid] = (frame_no, x, y)
            return False

        if (frame_no - a[0]) / self.fps >= self.max_seconds:
            self.flagged.add(tid)
            return True
        return False
