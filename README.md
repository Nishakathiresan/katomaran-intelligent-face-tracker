# Katomaran Intelligent Face Tracker

An intelligent face tracking system that detects, tracks, recognizes, and logs people from video input.

## Project Overview

This project processes a video and identifies human faces using a combination of:

- YOLO for face detection
- ByteTrack for multi-object tracking
- InsightFace for face embeddings
- Cosine similarity for face recognition
- SQLite for storing face embeddings and entry/exit events

The system can recognize previously registered faces and register new faces automatically.

## System Workflow

Video Input
    ↓
Face Detection
    ↓
Face Tracking
    ↓
Face Embedding
    ↓
Face Recognition
    ↓
Face Registration / Matching
    ↓
Entry and Exit Event Logging
    ↓
SQLite Database

## Features

- Face detection from video
- Multi-face tracking
- Face embedding generation
- Face recognition using embedding similarity
- Automatic registration of new faces
- Duplicate entry prevention
- Duplicate exit prevention
- Entry and exit event logging
- SQLite database storage
- Configurable detection and recognition settings
- Application logging

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Video processing |
| YOLO / Ultralytics | Face detection |
| Supervision | Object tracking |
| InsightFace | Face embeddings |
| ONNX Runtime | InsightFace inference |
| NumPy | Numerical operations |
| SQLite | Face and event storage |

## Project Structure

```text
face-tracker/
│
├── src/
│   ├── database.py
│   ├── detector.py
│   ├── embedder.py
│   ├── logger.py
│   ├── pipeline.py
│   └── tracker.py
│
├── data/
│   └── visitors.db
│
├── models/
│   └── yolov8n-face.pt
│
├── videos/
│   └── sample1.mp4
│
├── logs/
│
├── config.json
├── main.py
├── requirements.txt
├── .gitignore
└── README.md