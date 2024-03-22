Transportation Detection and Tracking CLI

This Python package, track_transportation, provides functionalities for detecting and tracking objects related to transportation in videos.

Installation
To install track_transportation, you can use pip:

bash
pip install track_transportation


Usage:
Once installed, you can use the provided command-line interface (CLI) to detect and track transportation objects in a video.

Detection Function:
The detect_objects function in the detection.py module allows you to detect transportation objects in an image. Here's how you can use it in your Python script:

python

    from transportation.detection import detect_objects
    import cv2

    <!-- Load image -->

    image = cv2.imread('image.jpg')

    <!-- Perform object detection -->

    detections = detect_objects(image)

    <!-- Process detections as needed -->

    print("Detected objects:")
    print(detections)


Requirements
This package requires the following dependencies:

1. numpy
2. opencv-python
3. torch
4. matplotlib
5. ultralytics
6. cvzone
7. sort
8. tqdm

These dependencies will be automatically installed when you install track_transportation.

Notes
1. Ensure that you have a compatible version of Python installed on your system.
2. It's recommended to use a virtual environment to manage your Python dependencies.
3. Make sure that the input video file exists and is accessible to the CLI.