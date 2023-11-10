import os

import cv2

video_num = 2
path_to_video = f"/Users/arkadiysotnikov/PycharmProjects/astute-vision-retail-tracking/data/{video_num}.mp4"
# extract 60th frame
cap = cv2.VideoCapture(path_to_video)
while cap.isOpened():
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    success, frame = cap.read()
    os.makedirs(f"/Users/arkadiysotnikov/PycharmProjects/astute-vision-retail-tracking/data/{video_num}", exist_ok=True)
    cv2.imwrite(f"/Users/arkadiysotnikov/PycharmProjects/astute-vision-retail-tracking/data/{video_num}/{frame_number}.png", frame)