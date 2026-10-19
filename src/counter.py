class LineCounter:
    """Counts each tracked object once when it crosses a horizontal line."""

    def __init__(self, line_y):
        self.line_y = line_y
        self.last_y = {}
        self.counted = set()
        self.totals = {}

    def update(self, track_id, label, cy):
        prev = self.last_y.get(track_id)
        self.last_y[track_id] = cy

        if prev is None or track_id in self.counted:
            return None

        crossed_down = prev < self.line_y <= cy
        crossed_up = prev > self.line_y >= cy
        if not (crossed_down or crossed_up):
            return None

        self.counted.add(track_id)
        self.totals[label] = self.totals.get(label, 0) + 1
        return "down" if crossed_down else "up"
