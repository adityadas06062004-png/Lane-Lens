import os
import tempfile

import cv2
import pandas as pd
import streamlit as st

import storage
from detector import DEFAULTS, PROFILES, draw_setup, process_video

st.set_page_config(page_title="LaneLens", layout="wide")
st.title("LaneLens")
st.caption("Traffic rule violation detection with a source check and a human review step")

ICONS = {"ok": "PASS", "warn": "WARN", "fail": "FAIL", "skipped": "SKIPPED"}


@st.cache_data
def save_upload(name, data):
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


upload = st.file_uploader("Upload a clip from a fixed traffic camera", type=["mp4", "mov", "avi"])
if not upload:
    st.info("Only footage recorded by a real fixed camera is accepted. Generated, edited "
            "or handheld clips are rejected by the source check.")
    st.stop()

src = save_upload(upload.name, upload.getvalue())
cap = cv2.VideoCapture(src)
ok, first = cap.read()
cap.release()
if not ok:
    st.error("Could not read that video.")
    st.stop()

cfg = dict(DEFAULTS)

with st.sidebar:
    st.header("Footage")
    declared = st.text_input("Source (camera ID, place, date)")
    attested = st.checkbox("I confirm this is original, unedited footage from a fixed camera")

    st.header("Detection")
    cfg["profile"] = st.radio("Profile", list(PROFILES), format_func=lambda p: p.replace("_", " "),
                              help="Fast traffic uses a larger model and a tracker built for big jumps between frames.")
    cfg["conf"] = st.slider("Detection confidence", 0.15, 0.8, PROFILES[cfg["profile"]]["conf"], 0.05)

    st.header("Rules")
    cfg["rules"] = st.multiselect("Active rules", ["red_light", "wrong_way", "illegal_stop", "speeding"],
                                  DEFAULTS["rules"])
    flow = st.selectbox("Normal traffic flow", ["auto", "down", "up"],
                        format_func=lambda v: {"auto": "Learn from traffic (any road layout)",
                                               "down": "One way, towards the bottom",
                                               "up": "One way, towards the top"}[v])
    cfg["traffic_flow"] = flow
    cfg["signal_direction"] = st.radio("Signal controls traffic moving", ["down", "up"])

    st.header("Scene setup")
    cfg["stop_line"] = st.slider("Stop line height", 0.1, 0.95, 0.6, 0.01)
    sx1 = st.slider("Signal box left", 0.0, 1.0, 0.45, 0.01)
    sx2 = st.slider("Signal box right", 0.0, 1.0, 0.5, 0.01)
    sy1 = st.slider("Signal box top", 0.0, 1.0, 0.05, 0.01)
    sy2 = st.slider("Signal box bottom", 0.0, 1.0, 0.15, 0.01)
    cfg["signal_roi"] = [min(sx1, sx2), min(sy1, sy2), max(sx1, sx2), max(sy1, sy2)]
    cfg["max_dwell_seconds"] = st.number_input("Max seconds stopped in zone", 2, 120, 8)

    st.header("Speed")
    cfg["speed_limit_kmh"] = st.number_input("Limit (km/h)", 10, 200, 40)
    with st.expander("Calibrate the road"):
        st.caption("Mark a rectangle on the road with a known size, such as a lane width "
                   "by the gap between two markings. Match the cyan shape in the preview to it.")
        corners = []
        for label, dx, dy in [("Top left", 0.40, 0.45), ("Top right", 0.60, 0.45),
                              ("Bottom right", 0.80, 0.90), ("Bottom left", 0.20, 0.90)]:
            cx = st.slider(f"{label} x", 0.0, 1.0, dx, 0.01)
            cy = st.slider(f"{label} y", 0.0, 1.0, dy, 0.01)
            corners.append([cx, cy])
        width_m = st.number_input("Real width of the rectangle (m)", 1.0, 50.0, 7.0)
        length_m = st.number_input("Real length of the rectangle (m)", 1.0, 200.0, 10.0)
        if "speeding" in cfg["rules"]:
            cfg["speed_plane"] = {"corners": corners, "width_m": width_m, "length_m": length_m}

st.subheader("Check your setup")
st.image(cv2.cvtColor(draw_setup(first, cfg), cv2.COLOR_BGR2RGB), width="stretch",
         caption="Red line: stop line. Yellow box: signal. Blue shape: no stopping zone. "
                 "Cyan shape: speed calibration rectangle.")

if st.button("Run detection", type="primary", disabled=not attested):
    bar = st.progress(0.0)
    out = process_video(src, cfg, out_dir="runs", on_progress=bar.progress,
                        declared_source=declared)
    st.session_state["run"] = out

if not attested:
    st.caption("Tick the confirmation box in the sidebar to enable the run.")

out = st.session_state.get("run")
if not out:
    st.stop()

check = out["authenticity"]
st.subheader("Source check")
verdict = check["verdict"]
if verdict == "block":
    st.error("Rejected. This clip does not behave like footage from a real fixed camera.")
elif verdict == "review":
    st.warning("Unverified. Results are marked as unverified and need a closer look.")
else:
    st.success("Passed the automatic checks. This does not prove the clip is genuine.")

st.table(pd.DataFrame([{"Check": c["name"], "Result": ICONS[c["status"]], "Detail": c["detail"]}
                       for c in check["checks"]]))
st.code(f"SHA-256  {check['sha256']}", language=None)

for note in out["warnings"]:
    st.info(note)

if out["video"] is None:
    st.stop()

db = storage.connect(os.path.join("runs", "violations.db"))
rows = storage.for_run(db, out["run_id"])

st.subheader("Result")
st.video(out["video"])

if not rows:
    st.success("No violations found with these settings.")
    st.stop()

frame = pd.DataFrame(rows)
st.bar_chart(frame["rule"].value_counts())

st.subheader("Review queue")
st.caption("Confirm or reject each detection. This gives you a real precision number.")

options = ["pending", "confirmed", "false_alarm"]
for row in rows:
    left, right = st.columns([1, 2])
    left.image(row["evidence"], width="stretch")
    with right:
        st.markdown(f"**{row['rule'].replace('_', ' ')}** at {row['video_second']}s")
        st.write(f"{row['vehicle']} #{row['track_id']}, {row['detail']}")
        choice = st.selectbox("Review", options, options.index(row["review"]),
                              key=f"review_{row['id']}")
        if choice != row["review"]:
            storage.set_review(db, row["id"], choice)

score = storage.precision(db, out["run_id"])
if score is not None:
    st.metric("Precision on reviewed detections", f"{score:.0%}")

st.download_button("Download violations CSV",
                   frame.drop(columns=["evidence"]).to_csv(index=False), "violations.csv")
