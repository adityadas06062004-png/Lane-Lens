import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    rule TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    vehicle TEXT NOT NULL,
    video_second REAL NOT NULL,
    detail TEXT,
    evidence TEXT,
    review TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL
)
"""


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute(SCHEMA)
    db.commit()
    return db


def add(db, run_id, rule, track_id, vehicle, video_second, detail, evidence):
    cur = db.execute(
        "INSERT INTO violations (run_id, rule, track_id, vehicle, video_second,"
        " detail, evidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, rule, track_id, vehicle, round(video_second, 2),
         detail, evidence, time.time()),
    )
    db.commit()
    return cur.lastrowid


def for_run(db, run_id):
    rows = db.execute(
        "SELECT * FROM violations WHERE run_id = ? ORDER BY video_second", (run_id,)
    )
    return [dict(r) for r in rows]


def set_review(db, violation_id, status):
    db.execute("UPDATE violations SET review = ? WHERE id = ?", (status, violation_id))
    db.commit()


def precision(db, run_id=None):
    sql = "SELECT review, COUNT(*) AS n FROM violations WHERE review != 'pending'"
    args = ()
    if run_id:
        sql += " AND run_id = ?"
        args = (run_id,)
    sql += " GROUP BY review"
    counts = {r["review"]: r["n"] for r in db.execute(sql, args)}
    good = counts.get("confirmed", 0)
    bad = counts.get("false_alarm", 0)
    if good + bad == 0:
        return None
    return good / (good + bad)
