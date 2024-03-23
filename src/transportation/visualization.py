# visualization.py
import cv2
import cvzone
import numpy as np
previous_speed_ms = 0 
totalCount = []
object_paths = {}  # Dictionary to store object paths
vehicles_entering = {}  # Keep track of when vehicles enter area_3
vehicles_elapsed_time = {}  # Keep track of elapsed time for each vehicle
import time


def visualize_object_paths(img, object_paths,resultsTracker,area_1, area_2, area_3):
   
    for area in [area_1 , area_2, area_3]:
            cv2.polylines(img, [np.array(area, np.int32)], True, (0,0,255), 1)

    for result in resultsTracker:
            x1, y1, x2, y2, track_id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 4, (0,0,0))
            cvzone.cornerRect(img, (x1, y1, w, h), l=1, rt=1, colorR=(0,0,0))
            if track_id not in object_paths:
                object_paths[track_id] = []
                object_paths[track_id].append((cx, cy))  # Assuming cx, cy are the coordinates of the object

                # Draw path on image
                if len(object_paths[track_id]) > 1:
                    for i in range(1, len(object_paths[track_id])):
                        cv2.line(img, object_paths[track_id][i-1], object_paths[track_id][i], (0,0,255), thickness=3)

                for area in [area_1, area_2, area_3]:
                    if cv2.pointPolygonTest(np.array(area, np.int32), (int(cx), int(cy)), False) >= 0:
                        if track_id not in totalCount:
                            vehicles_entering[track_id] = time.time()
                            totalCount.append(track_id)

                        # Calculate average speed
                        if track_id in vehicles_entering and area != area_1:
                            elapsed_time = time.time() - vehicles_entering[track_id]

                            if elapsed_time != 0:  # Avoid division by zero
                                if track_id not in vehicles_elapsed_time:
                                    vehicles_elapsed_time[track_id] = elapsed_time

                                if track_id in vehicles_elapsed_time:
                                    elapsed_time = vehicles_elapsed_time[track_id]
                                    distance = 18  # meters
                                    a_speed_ms = distance / elapsed_time
                                    a_speed_kh = a_speed_ms * 3.6

                                    # Calculating acceleration
                                    if elapsed_time != 0:  # Avoid division by zero
                                        acceleration = (a_speed_ms - previous_speed_ms) / elapsed_time
                                    else:
                                        acceleration = 0

                                    previous_speed_ms = a_speed_ms

                                    # Print or display velocity and acceleration on the box
                                    velocity_text = f'Spd: {a_speed_kh:.1f} km/h'
                                    acceleration_text = f'Acc: {acceleration:.2f} m/s²'
                                    cv2.putText(img, velocity_text, (max(0, x1+w//4), max(20, y1-14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA | cv2.FONT_ITALIC)
                                    cv2.putText(img, acceleration_text, (max(0, x1+w//4), max(20, y1-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA | cv2.FONT_ITALIC)
    return img
