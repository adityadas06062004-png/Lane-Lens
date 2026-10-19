# LaneLens project context

Owner: Aditya Das, third year B.Tech CSE, preparing for FAANG placements.

Goal: a traffic rule violation detector for his CV, built on a zero budget and good enough to talk about in interviews. Code and text should read as human written, with no unnecessary dashes.

Stack: Python, YOLOv8, OpenCV, SQLite, Streamlit, FFmpeg, Docker, GitHub Actions.

Structure:
- src/authenticity.py source check (cuts, fixed camera, metadata, C2PA, SHA-256)
- src/rules.py rules, flow field, ground plane calibration, signal reader
- src/counter.py line crossing counter
- src/storage.py SQLite runs, violations and precision
- src/detector.py pipeline with standard and fast_traffic profiles
- src/fast_bytetrack.yaml tracker settings for fast traffic
- src/app.py Streamlit app
- tests/ unit tests and a pipeline test with a fake model

Decisions so far:
- Rules work on the bottom centre of each vehicle box
- One violation per vehicle per rule
- Wrong way is learned per grid cell from traffic, so driving side does not matter
- Speeding is off until the road is calibrated with a four point ground plane
- Clips that fail the source check are rejected, not analysed
- Every detection gets human review, which gives a precision number for the README and CV

Test results from his machine (first version):
- traffic.mp4 flagged every car as wrong way because the old rule assumed one direction
- traffic3.mp4 gave speeds of 151 km/h from an uncalibrated pixel scale
Both are fixed in this version.

Status: 24 tests pass. The full pipeline has been tested with a fake model, and needs to be run on real footage.
