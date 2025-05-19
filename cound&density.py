import cv2
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load video or webcam
cap = cv2.VideoCapture("crowd_video.mp4")  # Use 0 for webcam

# Grid size for density map (e.g., 4x4)
GRID_ROWS, GRID_COLS = 4, 4

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Detect people
    results = model(frame)
    detections = results.pandas().xyxy[0]
    people = detections[detections['name'] == 'person']
    crowd_count = len(people)

    # Draw boxes
    for _, person in people.iterrows():
        x1, y1, x2, y2 = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Generate density matrix
    density_matrix = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    for _, person in people.iterrows():
        cx = int((person['xmin'] + person['xmax']) / 2)
        cy = int((person['ymin'] + person['ymax']) / 2)
        grid_x = min(cx * GRID_COLS // width, GRID_COLS - 1)
        grid_y = min(cy * GRID_ROWS // height, GRID_ROWS - 1)
        density_matrix[grid_y][grid_x] += 1

    # Convert density to heatmap (resized to frame size)
    heatmap = cv2.applyColorMap(
        cv2.resize((density_matrix * 20).astype(np.uint8), (width, height)),
        cv2.COLORMAP_JET
    )
    blended = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

    # Display crowd count
    cv2.putText(blended, f"Estimated Crowd: {crowd_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Crowd Density Estimation", blended)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()