# LaneLens project context

Owner: Aditya Das, third year B.Tech CSE, preparing for FAANG placements.

Goal: a traffic rule violation detector for his CV, built on a zero budget and good enough to talk about in interviews.

Stack: Python, YOLOv8n, OpenCV, SQLite, Streamlit, FFmpeg, Docker, GitHub Actions.

Structure:
- src/counter.py line crossing counter
- src/rules.py red light, wrong way, speeding, illegal stop rules and the signal reader
- src/storage.py SQLite storage and the precision calculation
- src/detector.py video pipeline
- src/app.py Streamlit app with scene setup and a review queue
- tests/ unit tests for the rules and counter

Decisions so far:
- Rules work on the bottom centre of each vehicle box
- One violation per vehicle per rule
- Every detection needs human review, which gives a precision number for the README and CV

Status: core pipeline written and unit tested. Not yet run on real footage.
