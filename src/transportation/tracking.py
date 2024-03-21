# tracking.py
import numpy as np
from sort import Sort

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

def track_objects(detections):
    return tracker.update(detections)
