import cv2
path_to_video = "/Users/arkadiysotnikov/PycharmProjects/astute-vision-retail-tracking/data/1.mp4"
# extract 60th frame
cap = cv2.VideoCapture(path_to_video)
while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < 60:
    success, frame = cap.read()
cv2.imwrite("/Users/arkadiysotnikov/PycharmProjects/astute-vision-retail-tracking/data/1.png", frame)