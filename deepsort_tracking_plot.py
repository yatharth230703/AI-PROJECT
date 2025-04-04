import cv2
import numpy as np
import sys
from collections import defaultdict
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO(r'C:\Users\Yatharth\Desktop\desktop1\AI\AIMS-Research-Proj\yolov8n-face.pt')  
# Initialize Deep SORT
max_cosine_distance = 0.4
nn_budget = None
#appearance descriptor model
model_filename = r'C:\Users\Yatharth\Desktop\desktop1\AI\AIMS-Research-Proj\mars-small128.pb'  
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

#NMS
def non_max_suppression(boxes, scores, threshold=0.3):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=threshold)
    return indices

cap = cv2.VideoCapture(r"C:\Users\Yatharth\Desktop\desktop1\AI\AIMS-Research-Proj\final_test_file.mp4")

# Store the track history
track_history = defaultdict(lambda: [])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    height, width, _ = frame.shape

    # YOLOv8 detection
    results = yolo_model(frame)
    detections_data = results[0].boxes.data.cpu().numpy()  

    yolo_boxes = []
    yolo_scores = []

    for detection in detections_data:
        confidence = detection[4]
        if confidence > 0.5:
            x1, y1, x2, y2 = detection[:4]
            box_width = x2 - x1
            box_height = y2 - y1
            yolo_boxes.append([int(x1), int(y1), int(box_width), int(box_height)])
            yolo_scores.append(float(confidence))

    #NMS
    indices = non_max_suppression(yolo_boxes, yolo_scores)
    detections = []

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = yolo_boxes[i]
            bbox = [x, y, w, h]
            detections.append(Detection(bbox, yolo_scores[i], encoder(frame, [bbox])[0]))


    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        track_id = track.track_id

    
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        center_x, center_y = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
        track_history[track_id].append((center_x, center_y))

        points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

        
        if len(track_history[track_id]) > 50:  
            track_history[track_id] = track_history[track_id][-50:]

    
    cv2.imshow('YOLOv8 Face Detection and Deep SORT Tracking with Movement Plotting', frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
