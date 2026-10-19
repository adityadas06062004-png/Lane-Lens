import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from counter import LineCounter


def test_counts_downward_crossing():
    c = LineCounter(line_y=100)
    assert c.update(1, "car", 90) is None
    assert c.update(1, "car", 110) == "down"
    assert c.totals == {"car": 1}


def test_counts_upward_crossing():
    c = LineCounter(line_y=100)
    c.update(2, "bus", 120)
    assert c.update(2, "bus", 95) == "up"


def test_same_object_counted_once():
    c = LineCounter(line_y=100)
    c.update(3, "car", 90)
    c.update(3, "car", 110)
    c.update(3, "car", 90)
    assert c.update(3, "car", 110) is None
    assert c.totals["car"] == 1


def test_no_crossing_no_count():
    c = LineCounter(line_y=100)
    c.update(4, "person", 20)
    assert c.update(4, "person", 60) is None
    assert c.totals == {}
