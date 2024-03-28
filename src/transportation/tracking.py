# # tracking.py
# import numpy as np
# from transportation.sort import Sort

# tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# def track_objects(detections):
#     return tracker.update(detections)


# tracking.py

import numpy as np
from transportation.sort import Sort

class ObjectTracker:
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    def track_objects(self, detections):
        return self.tracker.update(detections)

    # def run_example(self):

    #     tracked_objects = self.track_objects(detections)

    #     # Do something with the tracked objects, e.g., print them
    #     print("Tracked objects:", tracked_objects)