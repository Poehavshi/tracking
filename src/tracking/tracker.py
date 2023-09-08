from ultralytics import YOLO


class DirectionPredictor:
    def __init__(self, yolo_version="yolov8x.pt"):
        self.tracker = YOLO(yolo_version)

    def predict(self, frames):
        person_classes = [0]
        # fixme now it predict only for the first stream
        results = self.tracker.track(frames[0], persist=True, classes=person_classes)
    def __call__(self, frames):
        self.predict(frames)
