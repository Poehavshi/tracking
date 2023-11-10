from collections import defaultdict
import json

import cv2
import numpy as np
from ultralytics import YOLO

# init the YOLOv8 model
model = YOLO('yolov8n.pt')


# create two video captures for each camera
def open_annotations(path_to_annotations):
    with open(path_to_annotations) as file:
        zones_json = json.load(file)
        return {shape["label"]: np.array(shape["points"], dtype=np.int32) for shape in zones_json["shapes"]}


directory_with_videos = "/Users/arkadiysotnikov/PycharmProjects/astute-vision-retail-tracking/data"
video_captures = [cv2.VideoCapture(f"{directory_with_videos}/{i}.mp4") for i in range(1, 3)]
zones = [open_annotations(f"{directory_with_videos}/{i}.json") for i in range(1, 3)]


while video_captures[1].isOpened() and video_captures[1].get(cv2.CAP_PROP_POS_FRAMES) < 30:
    success, frame = video_captures[1].read()


# Store the track history
track_history = defaultdict(lambda: [])
flag = False
# Loop over the frames of the all videos
while True:
    success, frames = zip(*[cap.read() for cap in video_captures])
    if all(success):
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frames, persist=True, classes=[0])
        for i, camera_result in enumerate(results):
            if camera_result.boxes is None or camera_result.boxes.id is None:
                continue
            # draw bbox with track id
            boxes = camera_result.boxes.xywh.cpu()
            track_ids = camera_result.boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                # draw bbox
                cv2.rectangle(frames[i], (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
                # get the down left corner
                x_down_left = x - w / 2
                y_down_left = y - h / 2
                # draw x_down_left, y_down_left
                cv2.circle(frames[i], (int(x_down_left), int(y_down_left)), 3, (0, 255, 0), -1)
                track = track_history[track_id]
                track.append((float(x_down_left), float(y_down_left)))
                if len(track) > 30:
                    track.pop(0)
                if len(track) > 2:
                    dx = track[-1][0] - track[-2][0]
                    dy = track[-1][1] - track[-2][1]
                    direction = np.arctan2(dy, dx)
                    velocity = np.sqrt(dx ** 2 + dy ** 2)
                    if velocity < 1.3:
                        continue
                    cv2.arrowedLine(frames[i], (int(x), int(y)),
                                    (int(x + 2 * 30 * np.cos(direction)), int(y + 2 * 30 * np.sin(direction))),
                                    (0, 255, 0), 2)
                    cv2.putText(frames[i], str(track_id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            for zone_name, points in zones[i].items():
                points = np.array(points, dtype=np.int32)
                # if person is in the zone, draw it in green
                if cv2.pointPolygonTest(points, (int(x), int(y)), False) >= 0:
                    cv2.polylines(frames[i], [points], isClosed=True, color=(0, 255, 0), thickness=2)
                    # draw text that shows the zone name in the top left corner of the frame
                    cv2.putText(frames[i], zone_name, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    if flag:
                        cv2.putText(frames[i], "On the left side", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    else:
                        cv2.putText(frames[i], "Behind", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        # # Display the annotated frame
        cv2.imshow("Astute Vision 1", frames[0])
        cv2.imshow("Astute Vision 2", frames[1])

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
for cap in video_captures:
    cap.release()
cv2.destroyAllWindows()
