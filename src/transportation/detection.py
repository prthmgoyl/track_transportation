# detection.py
import numpy as np
from ultralytics import YOLO
import cv2
import math

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

def detect_objects(img,stream=False , objects=None , box=False , segment=False , conf=0.3, text=True):

    if objects is None:
        objects = classNames


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
    
    if segment:
        segmented_images = segment_objects(results)
        for i, segmented_image in enumerate(segmented_images):
            img = cv2.addWeighted(img, 1, segmented_image, 0.5, 0)
            #cv2.putText(img, f'Segment {i}', (10, 35 * (i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA | cv2.FONT_ITALIC)

    if box:
        bounding_boxes = detect_boxes(results,objects,conf)
        for box in bounding_boxes:
            x1, y1, x2, y2 = box['coordinates']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(img, f"{box['class']} {box['confidence']}", (x1, max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA | cv2.FONT_ITALIC)

    return img
    
# def detection(results):
    
#     return detections

def segment_objects(results): 
    segmented_images = []
    for r in results:
        img = r.imgs[0]  # Extracting the image
        masks = r.pred[0]['masks']  # Extracting masks from the prediction
        for mask in masks:
            # Applying the mask to the image
            masked_image = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
            segmented_images.append(masked_image)
    return segmented_images

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
