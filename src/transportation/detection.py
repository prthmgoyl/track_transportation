# detection.py
import numpy as np
from ultralytics import YOLO
import cv2
import math
import tqdm
import cvzone
import time
from transportation.tracking import ObjectTracker

previous_speed_ms = 0 

tracker = ObjectTracker()

totalCount = []
object_paths = {} 
vehicles_entering = {}  # Keep track of when vehicles enter area_3
vehicles_elapsed_time = {}  # Keep track of elapsed time for each vehicle
area_1 = [(735, 203), (540, 208),(260, 605), (980, 569)]
area_2 = [(645, 575), (960, 559), (820, 350), (650, 350)]
area_3 = [(650, 343), (655, 213), (545, 213), (450, 352)]


model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

def detect_objects(path,stream=True,csv=False,save=False , 
                   objects=None , box=False , segment=False , conf=0.3, text=True , track=False):
    
    cap = cv2.VideoCapture(path)

    #To save output
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_video_path.mp4', fourcc, 25, (int(cap.get(3)), int(cap.get(4))))
    
    #Classes
    if objects is None:
        objects = classNames
    
    # CSV file setup
    if csv:
        csv_file_path = 'data.csv'
        csv_columns = ['Frame Number', 'Object Type','Track ID']

    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_number in tqdm(range(total_frames)):
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=stream)
        detections = np.empty((0, 5))

       

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                conff = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass in objects and conff > conf:
                    currentArray = np.array([x1, y1, x2, y2, conff])
                    detections = np.vstack((detections, currentArray))
                    if text:
                     cv2.putText(img, currentClass , (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA | cv2.FONT_ITALIC)
        
        

        if track:
          cv2.polylines(img, [np.array(area_1, np.int32)], True, (255,255,255), 2)
          for area in [area_2, area_3]:
            cv2.polylines(img, [np.array(area, np.int32)], True, (0,0,255), 1)

          resultsTracker = tracker.update(detections)
          for result in resultsTracker:
            x1, y1, x2, y2, track_id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 4, (0,0,0))
            cvzone.cornerRect(img, (x1, y1, w, h), l=1, rt=1, colorR=(0,0,0))

            # Store object coordinates for tracing path
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



        # if segment:
        #     segmented_images = segment_objects(results)
        #     for i, segmented_image in enumerate(segmented_images):
        #         img = cv2.addWeighted(img, 1, segmented_image, 0.5, 0)
        #         #cv2.putText(img, f'Segment {i}', (10, 35 * (i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA | cv2.FONT_ITALIC)

        if box:
            bounding_boxes = detect_boxes(results,objects,conf)
            for box in bounding_boxes:
                x1, y1, x2, y2 = box['coordinates']
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.putText(img, f"{box['class']} {box['confidence']}", (x1, max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA | cv2.FONT_ITALIC)

    

# def segment_objects(results): 
#     segmented_images = []
#     for r in results:
#         img = r.imgs[0]  # Extracting the image
#         masks = r.pred[0]['masks']  # Extracting masks from the prediction
#         for mask in masks:
#             # Applying the mask to the image
#             masked_image = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
#             segmented_images.append(masked_image)
#     return segmented_images

def detect_boxes(results,objects,conf):
    bounding_boxes = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conff = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass in objects and conff > conf:
                bounding_boxes.append({
                    'class': currentClass,
                    'confidence': conff,
                    'coordinates': (x1, y1, x2, y2)
                })
    return bounding_boxes
