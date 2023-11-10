import json
import math
import os
from collections import defaultdict
from logging import getLogger

import cv2
import numpy as np
import torch
import torchreid
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


logger = getLogger(__name__)
video_capture = cv2.VideoCapture("/Users/arkadiysotnikov/PycharmProjects/astute-vision-retail-tracking/examples/1/1.mp4")
video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = video_capture.get(cv2.CAP_PROP_FPS)
video_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter("/Users/arkadiysotnikov/PycharmProjects/astute-vision-retail-tracking/examples/1/1_out.mp4", video_fourcc, video_fps, (600, 1280))


def preprocess_frame(frame, need=False):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 128), antialias=None),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    preprocessed_frame = preprocess(frame)
    return preprocessed_frame.unsqueeze(0)


def open_annotations(path_to_annotations):
    with open(path_to_annotations) as file:
        zones_json = json.load(file)
        return {shape["label"]: np.array(shape["points"], dtype=np.int32) for shape in zones_json["shapes"]}


class ReIDModel:
    def __init__(self, ideal_frames_directory: list[str],
                 weights_path: str = "/Users/arkadiysotnikov/PycharmProjects/ReId/osnet_ain_x1_0.pth"):
        model = torchreid.models.build_model(
            name='osnet_ain_x1_0',  # Replace with your model architecture
            num_classes=702,  # Replace with the number of classes in your dataset
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        torchreid.utils.load_pretrained_weights(model, weights_path)
        model.eval()
        self.model = model
        self.ideal_embeddings = self.get_ideal_embeddings(ideal_frames_directory)

    def get_ideal_embeddings(self, paths):
        embeddings = {}
        for id, path in paths:
            img = Image.open(path)
            data = np.array(img)
            pre_frame = preprocess_frame(data)
            with torch.no_grad():
                cur_embedding = self.model(pre_frame)
                if id in embeddings:
                    embeddings[id].append(cur_embedding)
                else:
                    embeddings[id] = [cur_embedding]
        return embeddings



class YoloTracker:
    def __init__(self, annotations_path: list[str], yolo_version: str = "yolov8n.pt", max_track_length: int = 30,
                 min_track_length: int = 2, std_velocity_threshold: float = 1.3):
        self.min_track_length = min_track_length
        self.max_track_length = max_track_length
        self.std_velocity_threshold = std_velocity_threshold
        self.camera_number2locations = [open_annotations(path) for path in annotations_path]
        self.model = YOLO(yolo_version)
        ideal_paths = []
        base_path = "/Users/arkadiysotnikov/PycharmProjects/ReId/ideal"
        for i in range(1, 2):
            for j in range(12):
                ideal_paths.append((i, os.path.join(base_path, str(i), str(j) + '.jpg')))
        self.reid = ReIDModel(ideal_frames_directory=ideal_paths)
        self.track_history = defaultdict(lambda: [])

        self.frame_count = 0

    def predict(self, frames, destination_coords: tuple[int, int]) -> tuple[float, tuple[int, int]]:
        yolo_person_class = 0
        results = self.model.track(frames, persist=True, classes=[yolo_person_class], verbose=False)
        mini_map_x = None
        next_mini_map_x = None
        track_id = -1
        for i, camera_result in enumerate(results):
            boxes = camera_result.boxes.xywh.cpu().tolist()
            if boxes is None or camera_result.boxes.id is None:
                logger.info(f"Camera {i} has no boxes")
                continue
            track_ids = camera_result.boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                cv2.rectangle(frames[i], (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
                x_down_left = x - w / 2
                y_down_left = y + h / 2
                direction_angle = self.estimate_person_direction(track_id, x_down_left, y_down_left)
                if direction_angle is not None:
                    next_x, next_y = x_down_left + 150 * math.cos(direction_angle), y_down_left + 150 * math.sin(direction_angle)
                    next_next_x, next_next_y = x_down_left + 300 * math.cos(direction_angle), y_down_left + 300 * math.sin(direction_angle)
                    cv2.arrowedLine(frames[i], (int(x_down_left), int(y_down_left)), (int(next_x), int(next_y)), (0, 255, 0), 2)
                else:
                    next_x = x_down_left
                    next_y = y_down_left
                    next_next_x = x_down_left
                    next_next_y = y_down_left
                for zone_name, zone_points in self.camera_number2locations[i].items():
                    if cv2.pointPolygonTest(zone_points, (int(x_down_left), int(y_down_left)), False) >= 0:
                        current_zone = zone_name
                        cv2.polylines(frames[i], [zone_points], True, (0, 0, 255), 2)
                        cv2.putText(frames[i], f"Zone {current_zone}", (int(x_down_left), int(y_down_left)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        mini_map_x, mini_map_y = map(int, current_zone.split("_"))
                    if cv2.pointPolygonTest(zone_points, (int(next_x), int(next_y)), False) >= 0:
                        next_mini_map_x, next_mini_map_y = map(int, zone_name.split("_"))
                    elif cv2.pointPolygonTest(zone_points, (int(next_next_x), int(next_next_y)), False) >= 0:
                        next_mini_map_x, next_mini_map_y = map(int, zone_name.split("_"))
                cropped_person = frames[i][int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
                real_id = -1
                if self.reid is not None:
                    cropped_person = preprocess_frame(cropped_person)
                    with torch.no_grad():
                        cur_embedding = self.reid.model(cropped_person)
                        for id, embeddings in self.reid.ideal_embeddings.items():
                            for embedding in embeddings:
                                if torch.cosine_similarity(cur_embedding, embedding) > 0.7:
                                    real_id = id
                                    logger.info(f"Track is {id}")
                cv2.putText(frames[i], f"Track {track_id} Reid {real_id}", (int(x_down_left), int(y_down_left) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Display the annotated frame
        cv2.imshow("Astute Vision 1", frames[0])
        cv2.imshow("Astute Vision 2", frames[1])
        # plot map with track
        TILE_SIZE = 120
        mini_map = np.zeros((TILE_SIZE*2, TILE_SIZE*3, 3), dtype=np.uint8)
        # add grid
        for i in range(3):
            cv2.line(mini_map, (i*TILE_SIZE, 0), (i*TILE_SIZE, TILE_SIZE*2), (255, 255, 255), 2)
        for i in range(2):
            cv2.line(mini_map, (0, i*TILE_SIZE), (TILE_SIZE*3, i*TILE_SIZE), (255, 255, 255), 2)
        # draw current position
        if mini_map_x is not None and mini_map_y is not None:
            cv2.circle(mini_map, (mini_map_x*TILE_SIZE-TILE_SIZE//2, mini_map_y*TILE_SIZE-TILE_SIZE//2), TILE_SIZE//4, (0, 0, 255), -1)
            cv2.putText(mini_map, f"{real_id}", (mini_map_x*TILE_SIZE-TILE_SIZE//2, mini_map_y*TILE_SIZE-TILE_SIZE//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if next_mini_map_x is not None:
                cv2.arrowedLine(mini_map, (mini_map_x*TILE_SIZE-TILE_SIZE//2, mini_map_y*TILE_SIZE-TILE_SIZE//2), (next_mini_map_x*TILE_SIZE-TILE_SIZE//2, next_mini_map_y*TILE_SIZE-TILE_SIZE//2), (0, 255, 0), 2)
        cv2.imshow("Mini map", mini_map)
        # concat all frames in one
        concat_frame = np.concatenate((frames[0], frames[1]), axis=1)
        # add black fill to the top with concat frame
        concat_frame = np.concatenate((np.zeros((TILE_SIZE*2, concat_frame.shape[1], 3), dtype=np.uint8), concat_frame), axis=0)
        # add mini map in black fill
        concat_frame[0:TILE_SIZE*2, concat_frame.shape[1]-TILE_SIZE*3:concat_frame.shape[1]] = mini_map
        cv2.imshow("Astute Vision", concat_frame)
        print(concat_frame.shape)
        # save image
        os.makedirs("output", exist_ok=True)
        self.frame_count = self.frame_count + 1
        cv2.imwrite(f"output/{self.frame_count}.jpg", concat_frame)
        return 0, (-1, -1)

    def predict_next_position(self, track_id, x_down_left, y_down_left):
        track = self.track_history[track_id]
        if len(track) > 1:
            dx = track[-1][0] - track[-2][0]
            dy = track[-1][1] - track[-2][1]
            person_direction_angle = np.arctan2(dy, dx)
            person_velocity = np.sqrt(dx ** 2 + dy ** 2)
            if person_velocity < self.std_velocity_threshold:
                person_velocity = 0
            if person_velocity > 0:
                next_x = x_down_left + person_velocity * math.cos(person_direction_angle)
                next_y = y_down_left + person_velocity * math.sin(person_direction_angle)
                return next_x, next_y
        return x_down_left, y_down_left

    def estimate_person_direction(self, track_id, x_down_left, y_down_left):
        track = self.track_history[track_id]
        track.append((float(x_down_left), float(y_down_left)))
        if len(track) > self.max_track_length:
            track.pop(0)
        if len(track) > self.min_track_length:
            dx = track[-1][0] - track[-2][0]
            dy = track[-1][1] - track[-2][1]
            person_direction = np.arctan2(dy, dx)

            person_velocity = np.sqrt(dx ** 2 + dy ** 2)
            if person_velocity < self.std_velocity_threshold:
                person_velocity = 0
            if person_velocity > 0:
                return person_direction
        return 0


IP_CAMERAS_URLS = ["examples/1/1.mp4", "examples/1/2.mp4"]
IP_CAMERAS_ANNOTATIONS = ["examples/1/1.json", "examples/1/2.json"]
video_captures = [cv2.VideoCapture(url) for url in IP_CAMERAS_URLS]

yolo_tracker = YoloTracker(IP_CAMERAS_ANNOTATIONS)
counter = 0
# Loop over the frames of the all videos
while True or counter < 300:
    counter += 1
    success, frames_from_cameras = zip(*[cap.read() for cap in video_captures])
    if not all(success):
        break

    # Apply the tracker to the frames
    yolo_tracker.predict(frames_from_cameras, (0, 0))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release the video capture object and close the display window
for cap in video_captures:
    cap.release()
cv2.destroyAllWindows()
video_writer.release()