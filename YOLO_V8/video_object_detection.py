import cv2 as cv
from yolov8 import YOLOv8
import sys

onnx_file_path = "./models/yolov8m.onnx"
yolo_model = YOLOv8(onnx_file_path, conf_thres=0.5, iou_thres=0.5)


test_video_path = "./images/video.mp4"
source = cv.VideoCapture(test_video_path)

width = int(source.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(source.get(cv.CAP_PROP_FRAME_HEIGHT))

# Create the video writer
out_video = cv.VideoWriter(
    "output_video.mp4", cv.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
)

while True:
    has_frame, frame = source.read()
    if not has_frame:
        break

    # Initialize YOLOV8 class
    yolo_model(frame)

    detected_objects = yolo_model.draw_detections(frame)

    out_video.write(detected_objects)

    # cv.imshow("YOLO", detected_objects)

source.release()
out_video.release()
cv.destroyAllWindows()
print("Vidoe has been created")
