Transportation Detection and Tracking CLI

This Python package, track_transportation, provides functionalities for detecting and tracking objects related to transportation in videos.

Installation
To install track_transportation, you can use pip:

bash
pip install track_transportation


Usage
Once installed, you can use the provided command-line interface (CLI) to detect and track transportation objects in a video.

Here's how you can use it:

bash
track_transportation <video_path>

Replace <video_path> with the path to your video file.

Example
Suppose you have a video file named traffic.mp4 that you want to analyze. You can run:

bash
track_transportation traffic.mp4


This command will process the video, detect transportation objects, track their movements, and visualize their paths. It will also output a new video with the detected objects and their paths annotated.

Requirements
This package requires the following dependencies:

numpy
opencv-python
torch
matplotlib
ultralytics
cvzone
sort
tqdm
These dependencies will be automatically installed when you install track_transportation.

Notes
Ensure that you have a compatible version of Python installed on your system.
It's recommended to use a virtual environment to manage your Python dependencies.
Make sure that the input video file exists and is accessible to the CLI.