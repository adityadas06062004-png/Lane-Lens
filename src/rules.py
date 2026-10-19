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


class GroundPlane:
    """Maps image points on the road surface to metres.

    The four corners are a rectangle on the road (for example a stretch of lane)
    whose real width and length are known. Using a homography removes most of the
    perspective error that a single metres per pixel value would have.
    """

    def __init__(self, pixel_corners, width_m, length_m):
        src = np.float32(pixel_corners)
        dst = np.float32([[0, 0], [width_m, 0], [width_m, length_m], [0, length_m]])
        self.matrix = cv2.getPerspectiveTransform(src, dst)

    def to_metres(self, x, y):
        point = np.float32([[[x, y]]])
        mapped = cv2.perspectiveTransform(point, self.matrix)[0][0]
        return float(mapped[0]), float(mapped[1])


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


class FixedFlow:
    """For a one way road where the direction is known up front."""

    def __init__(self, direction):
        self.vector = (0.0, 1.0) if direction == "down" else (0.0, -1.0)

    def direction(self, x, y):
        return self.vector

    def learn(self, x, y, ux, uy):
        pass


class FlowField:
    """Learns the normal direction of travel for each part of the frame.

    The frame is split into a grid. Every cell keeps the sum of the unit motion
    vectors of the vehicles that moved through it. A cell is trusted once it has
    enough samples and they mostly agree. On a two way road the left and right
    lanes simply end up with opposite directions, so nothing depends on which
    side of the road the country drives on.
    """

    def __init__(self, width, height, cols=12, rows=8, min_samples=40, coherence=0.6):
        self.width = width
        self.height = height
        self.cols = cols
        self.rows = rows
        self.min_samples = min_samples
        self.coherence = coherence
        self.sum_x = np.zeros((rows, cols))
        self.sum_y = np.zeros((rows, cols))
        self.count = np.zeros((rows, cols))

    def _cell(self, x, y):
        col = min(max(int(x / self.width * self.cols), 0), self.cols - 1)
        row = min(max(int(y / self.height * self.rows), 0), self.rows - 1)
        return row, col

    def direction(self, x, y):
        row, col = self._cell(x, y)
        n = self.count[row, col]
        if n < self.min_samples:
            return None
        sx, sy = self.sum_x[row, col], self.sum_y[row, col]
        length = math.hypot(sx, sy)
        if length == 0 or length / n < self.coherence:
            return None
        return sx / length, sy / length

    def learn(self, x, y, ux, uy):
        row, col = self._cell(x, y)
        self.sum_x[row, col] += ux
        self.sum_y[row, col] += uy
        self.count[row, col] += 1


class WrongWayRule:
    name = "wrong_way"

    def __init__(self, flow, fps, min_motion, min_travel, window=5,
                 confirm_seconds=0.6, opposing_dot=-0.6):
        self.flow = flow
        self.window = window
        self.min_motion = min_motion
        self.min_travel = min_travel
        self.opposing_dot = opposing_dot
        self.confirm = max(int(fps * confirm_seconds), 3)
        self.trail = {}
        self.score = {}
        self.travel = {}
        self.flagged = set()

    def check(self, tid, x, y):
        trail = self.trail.setdefault(tid, deque(maxlen=self.window + 1))
        trail.append((x, y))
        if tid in self.flagged or len(trail) <= self.window:
            return False

        x0, y0 = trail[0]
        dx, dy = x - x0, y - y0
        dist = math.hypot(dx, dy)
        if dist < self.min_motion:
            return False

        ux, uy = dx / dist, dy / dist
        normal = self.flow.direction(x, y)

        if normal is None:
            self.flow.learn(x, y, ux, uy)
            return False

        dot = ux * normal[0] + uy * normal[1]
        if dot <= self.opposing_dot:
            self.score[tid] = self.score.get(tid, 0) + 1
            self.travel[tid] = self.travel.get(tid, 0.0) + dist / self.window
            if self.score[tid] >= self.confirm and self.travel[tid] >= self.min_travel:
                self.flagged.add(tid)
                return True
        else:
            self.score[tid] = max(0, self.score.get(tid, 0) - 1)
            if dot > 0.3:
                # only learn from traffic that agrees, so one bad driver can't flip a lane
                self.flow.learn(x, y, ux, uy)
        return False


class SpeedRule:
    name = "speeding"

    def __init__(self, fps, ground, limit_kmh, window_seconds=1.0, max_kmh=250):
        self.fps = fps
        self.ground = ground
        self.limit = limit_kmh
        self.max_kmh = max_kmh
        self.window = max(int(fps * window_seconds), 3)
        self.history = {}
        self.speeds = {}
        self.flagged = set()

    def check(self, tid, x, y, frame_no):
        mx, my = self.ground.to_metres(x, y)
        trail = self.history.setdefault(tid, deque(maxlen=self.window + 1))

        if trail:
            last_frame, lx, ly = trail[-1]
            gap = (frame_no - last_frame) / self.fps
            if gap > 0 and math.hypot(mx - lx, my - ly) / gap * 3.6 > self.max_kmh:
                # the tracker jumped to another vehicle, so start the history over
                trail.clear()

        trail.append((frame_no, mx, my))
        if len(trail) <= self.window or tid in self.flagged:
            return False

        f0, x0, y0 = trail[0]
        mid_f, mid_x, mid_y = trail[len(trail) // 2]
        full = math.hypot(mx - x0, my - y0) / ((frame_no - f0) / self.fps) * 3.6
        half_time = (frame_no - mid_f) / self.fps
        half = math.hypot(mx - mid_x, my - mid_y) / half_time * 3.6 if half_time > 0 else 0

        self.speeds[tid] = full
        if min(full, half) > self.limit:
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
