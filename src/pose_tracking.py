from collections import defaultdict
import cv2
import numpy as np

from config import PATH_TO_VIDEO
from random import seed, randint

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n-pose.pt')

# Open the video file
cap = cv2.VideoCapture(PATH_TO_VIDEO)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
# Store the track history
track_history = defaultdict(lambda: [])
counter = 0

seed(1)
# generate colors
colors = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(28)]

flag = False
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if counter >= 164:
        break
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=[0])

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        keypoints = results[0].keypoints.xy.cpu()

        # plot results
        for i in range(len(keypoints)):
            for j in range(len(keypoints[i])):
                x, y = keypoints[i][j]
                cv2.circle(frame, (int(x), int(y)), 3, colors[j], -1)
        # estimate person orientation by keypoints 5 and 6
        for box, track_id in zip(boxes, track_ids):
        cv2.imshow("YOLOv8 Tracking", frame)
        counter += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
