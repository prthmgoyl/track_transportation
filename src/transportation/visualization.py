# visualization.py
import cv2

def visualize_object_paths(img, object_paths):
    for track_id, path in object_paths.items():
        if len(path) > 1:
            for i in range(1, len(path)):
                cv2.line(img, path[i-1], path[i], (0,0,255), thickness=3)

    return img
