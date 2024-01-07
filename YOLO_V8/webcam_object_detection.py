import cv2 as cv
from yolov8 import YOLOv8
import sys

onnx_file_path = "./models/yolov8m.onnx"
yolo_model = YOLOv8(onnx_file_path, conf_thres=0.5, iou_thres=0.5)

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]


win_name = "YOLO"
cv.namedWindow(win_name, cv.WINDOW_NORMAL)

source = cv.VideoCapture(s)


while cv.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break

    frame = cv.flip(frame, 1)

    # Initialize YOLOV8 class
    yolo_model(frame)

    detected_objects = yolo_model.draw_detections(frame)
    cv.imshow(win_name, detected_objects)

source.release()
cv.destroyAllWindows()
