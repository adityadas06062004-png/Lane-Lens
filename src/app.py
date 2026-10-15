import os
import tempfile

import cv2
import pandas as pd
import streamlit as st

import storage
from detector import DEFAULTS, draw_setup, process_video

st.set_page_config(page_title="LaneLens", layout="wide")
st.title("LaneLens")
st.caption("Traffic rule violation detection with a human review step")


@st.cache_data
def save_upload(name, data):
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


upload = st.file_uploader("Upload a clip from a fixed camera", type=["mp4", "mov", "avi"])
if not upload:
    st.info("Use footage from a fixed camera that you own or that is free to use.")
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
    st.header("Rules")
    cfg["rules"] = st.multiselect("Active rules", DEFAULTS["rules"], DEFAULTS["rules"])
    cfg["allowed_direction"] = st.radio("Allowed direction of travel", ["down", "up"])

    st.header("Scene setup")
    cfg["stop_line"] = st.slider("Stop line height", 0.1, 0.95, 0.6, 0.01)
    sx1 = st.slider("Signal box left", 0.0, 1.0, 0.45, 0.01)
    sx2 = st.slider("Signal box right", 0.0, 1.0, 0.5, 0.01)
    sy1 = st.slider("Signal box top", 0.0, 1.0, 0.05, 0.01)
    sy2 = st.slider("Signal box bottom", 0.0, 1.0, 0.15, 0.01)
    cfg["signal_roi"] = [min(sx1, sx2), min(sy1, sy2), max(sx1, sx2), max(sy1, sy2)]

    st.header("Speed")
    cfg["speed_limit_kmh"] = st.number_input("Limit (km/h)", 10, 200, 40)
    with st.expander("Calibrate"):
        px_len = st.number_input("Known length in pixels", 1.0, 5000.0, 200.0)
        real_len = st.number_input("Same length in metres", 0.5, 200.0, 10.0)
        cfg["metres_per_pixel"] = real_len / px_len
        st.caption(f"{cfg['metres_per_pixel']:.4f} m per pixel")

    cfg["max_dwell_seconds"] = st.number_input("Max seconds stopped in zone", 2, 120, 8)
    cfg["conf"] = st.slider("Detection confidence", 0.2, 0.8, 0.4, 0.05)

st.subheader("Check your setup")
st.image(cv2.cvtColor(draw_setup(first, cfg), cv2.COLOR_BGR2RGB),
         caption="Red line is the stop line, yellow box is the signal, blue shape is the no stopping zone")

if st.button("Run detection", type="primary"):
    bar = st.progress(0.0)
    run_id, video, rows, _ = process_video(src, cfg, out_dir="runs", on_progress=bar.progress)
    st.session_state["run"] = {"id": run_id, "video": video}

run = st.session_state.get("run")
if not run:
    st.stop()

db = storage.connect(os.path.join("runs", "violations.db"))
rows = storage.for_run(db, run["id"])

st.subheader("Result")
st.video(run["video"])

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
    left.image(row["evidence"], use_container_width=True)
    with right:
        st.markdown(f"**{row['rule'].replace('_', ' ')}** at {row['video_second']}s")
        st.write(f"{row['vehicle']} #{row['track_id']}, {row['detail']}")
        choice = st.selectbox("Review", options, options.index(row["review"]),
                              key=f"review_{row['id']}")
        if choice != row["review"]:
            storage.set_review(db, row["id"], choice)

score = storage.precision(db, run["id"])
if score is not None:
    st.metric("Precision on reviewed detections", f"{score:.0%}")

st.download_button("Download violations CSV", frame.drop(columns=["evidence"]).to_csv(index=False),
                   "violations.csv")
