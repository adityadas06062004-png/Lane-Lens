# LaneLens

LaneLens watches footage from a fixed traffic camera and flags rule violations. It first checks that the clip behaves like real camera footage, then detects and tracks vehicles, applies the rules, and saves an evidence frame for every violation. A person confirms or rejects each detection in a review queue, which gives a real precision figure for each run.

## What it detects

| Rule | How it works |
|------|--------------|
| Red light | Reads the lamp colour from a box you place around the signal and flags a vehicle that crosses the stop line on red |
| Wrong way | Learns the normal direction of each part of the road from the traffic itself, then flags vehicles that keep moving against it |
| Speeding | Converts movement to km/h using a calibrated patch of road, so perspective is handled |
| Illegal stop | Flags a vehicle that sits still inside a no stopping zone for too long |

## Stack
Python, YOLOv8, OpenCV, SQLite, Streamlit, FFmpeg, Docker

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

## Source check
Before any detection runs, the clip goes through a source check:

- Scene cuts. A real camera feed is continuous, so any cut rejects the clip.
- Fixed viewpoint. The background must stay in place. Handheld, dashcam and most generated clips fail this.
- File metadata. Warns if the file names a video generation tool.
- Content credentials. If the free `c2patool` is installed, signed C2PA data that marks the media as generated rejects the clip.
- A SHA-256 of the file is stored with the run so the footage can be matched later.

This cannot prove a clip is genuine, because reliable detection of generated video is an unsolved problem and metadata is easy to remove. It rejects footage that breaks what a real fixed camera produces. For real use, take footage straight from the camera feed and keep the hash.

## Road layouts and driving side
Wrong way detection does not assume a direction. The frame is divided into a grid, each cell learns the dominant direction of the vehicles passing through it, and a vehicle is flagged only when it keeps moving against a cell that has a clear, consistent direction. Right hand and left hand traffic, two way roads and sideways roads all work without any setting. For a known one way street you can fix the direction in the app.

## Fast traffic
The fast traffic profile uses YOLOv8s, a larger input size, a lower confidence threshold and a tracker that tolerates large jumps between frames. On Apple Silicon the app uses the GPU automatically.

## Speed calibration
Mark a rectangle on the road with a known width and length, such as a lane width by the gap between two lane markings. The app maps image points to metres with a perspective transform. Speeding stays off until this is done, and tracker jumps that imply impossible speeds are discarded.

## How it works
1. The source check runs and the clip is accepted, marked unverified or rejected.
2. YOLOv8 detects cars, buses, trucks and motorcycles.
3. A tracker gives each vehicle an ID that stays stable between frames.
4. The bottom centre of each box, where the tyres meet the road, is checked against the active rules.
5. Each violation is stored once per vehicle in SQLite with an annotated screenshot.
6. The review queue turns confirmed and rejected detections into a precision score.

## Evaluation
Fill this in from your own clips using the review queue.

| Clip | Length | Violations flagged | Confirmed | False alarms | Precision |
|------|--------|--------------------|-----------|--------------|-----------|
|      |        |                    |           |              |           |

| Machine | Profile | Resolution | FPS |
|---------|---------|------------|-----|
|         |         |            |     |

## Limitations
- The source check reduces the chance of analysing fake or edited footage but cannot guarantee a clip is real.
- Wrong way detection needs some traffic to learn from, so very quiet clips give few results.
- Speed is only as accurate as the calibration, and it works best on a flat, straight stretch.
- The signal reader depends on a steady camera and a clearly visible lamp.
- Crowded scenes can swap or lose vehicle IDs.
- This is a research prototype. Its output is not legal evidence and a person must always review it.

## Next steps
- Helmet and triple riding checks using a model fine tuned on a public dataset
- Per lane statistics
- Number plate reading with blurred plates by default
- Live RTSP input with signed, hashed recordings

## Data
Only use footage you recorded or footage that is free to use, such as Pexels or Pixabay clips and public datasets like UA-DETRAC.
