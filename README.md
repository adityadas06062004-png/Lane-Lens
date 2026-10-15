# LaneLens

LaneLens watches footage from a fixed traffic camera and flags rule violations. Every detection comes with a saved evidence frame and goes into a review queue, where a person confirms it or marks it as a false alarm. Those reviews give you a real precision figure for each run.

## What it detects

| Rule | How it works |
|------|--------------|
| Red light | Reads the lamp colour from a box you place around the signal, flags a vehicle that crosses the stop line while it is red |
| Wrong way | Flags a vehicle that travels a set distance against the allowed direction |
| Speeding | Converts pixel movement to km/h using a calibrated metres per pixel value |
| Illegal stop | Flags a vehicle that sits still inside a no stopping zone for too long |

## Stack
Python, YOLOv8 Nano, OpenCV, SQLite, Streamlit, FFmpeg, Docker

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

Command line:

```bash
python src/detector.py clip.mp4 config.example.json
```

Tests:

```bash
pytest
```

Docker:

```bash
docker build -t lanelens .
docker run -p 8501:8501 lanelens
```

## How it works
1. YOLOv8n detects cars, buses, trucks and motorcycles.
2. The built in tracker assigns each vehicle an ID that stays stable between frames.
3. The bottom centre of each box, where the tyres meet the road, is checked against the active rules.
4. Each violation is stored once per vehicle in SQLite along with an annotated screenshot.
5. The review queue turns confirmed and rejected detections into a precision score.

## Setting up a scene
Use a fixed camera. In the app, adjust the stop line, signal box and zone until the preview matches your footage. For speed, measure a real distance on the road, such as a lane marking gap or a parked car length, enter its pixel length and its real length, and the app works out metres per pixel.

## Evaluation
Fill this table in from your own clips using the review queue.

| Clip | Length | Violations flagged | Confirmed | False alarms | Precision |
|------|--------|--------------------|-----------|--------------|-----------|
|      |        |                    |           |              |           |

| Machine | Resolution | FPS |
|---------|------------|-----|
|         |            |     |

## Limitations
- Speed is only accurate near the calibrated stretch of road, because perspective changes the scale.
- The signal reader depends on a steady camera and a clearly visible lamp.
- Crowded scenes can swap or lose vehicle IDs, which can cause missed or doubled flags.
- Night footage lowers detection confidence.
- This is a research prototype. Its output is not legal evidence and a person should always review it.

## Next steps
- Helmet and triple riding checks using a model fine tuned on a public dataset
- Per lane statistics
- Number plate reading with blurred plates by default
- Live RTSP stream input

## Data
Only use footage you recorded or footage that is free to use, such as Pexels or Pixabay clips and public datasets like UA-DETRAC.
# Lane-Lens
