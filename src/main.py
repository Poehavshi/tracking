from collections import defaultdict
import json

import cv2
import numpy as np
from config import PATH_TO_VIDEO

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8x.pt')

# Open the video file
cap = cv2.VideoCapture(PATH_TO_VIDEO)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
video_writer = cv2.VideoWriter("/Users/arkadiysotnikov/PycharmProjects/astute-vision-retail-tracking/data/output.avi",
                               cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

# Store the track history
track_history = defaultdict(lambda: [])
counter = 0
with open("/Users/arkadiysotnikov/PycharmProjects/astute-vision-retail-tracking/data/first_frame.json") as f:
    first_frame = json.load(f)
    zones = {shape["label"] : shape["points"] for shape in first_frame["shapes"]}
flag = False
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    counter += 1
    if counter >= 164:
        break
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=[0])

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            # fixme Only plot the track for the person with ID 5 for this example
            if track_id != 5:
                continue
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)
            # get velocity of the track to get known direction
            if len(track) > 2:
                dx = track[-1][0] - track[-2][0]
                dy = track[-1][1] - track[-2][1]
                direction = np.arctan2(dy, dx)
                velocity = np.sqrt(dx ** 2 + dy ** 2)
                cv2.arrowedLine(frame, (int(x), int(y)),
                                (int(x + 2 * velocity * np.cos(direction)), int(y + 2 * velocity * np.sin(direction))),
                                (0, 0, 255), 2)
                direction_deg = np.rad2deg(direction)
                if velocity > 3 and (direction_deg > 45 and direction_deg < 135):
                    flag = True
            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            # draw bbox
            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 0, 255), 2)
            # draw ID
            cv2.putText(frame, str(track_id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw the zones
        for zone_name, points in zones.items():
            points = np.array(points, dtype=np.int32)
            # if person is in the zone, draw it in green
            if cv2.pointPolygonTest(points, (int(x), int(y)), False) >= 0:
                cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                # draw text that shows the zone name in the top left corner of the frame
                cv2.putText(frame, zone_name, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                if flag:
                    cv2.putText(frame, "On the left side", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Behind", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", frame)
        video_writer.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
video_writer.release()
