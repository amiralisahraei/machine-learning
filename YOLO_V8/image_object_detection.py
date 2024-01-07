import cv2 as cv
from yolov8 import YOLOv8
import matplotlib.pyplot as plt
import numpy as np

onnx_file_path = "./models/yolov8m.onnx"
yolo_model = YOLOv8(onnx_file_path, conf_thres=0.5, iou_thres=0.5)


def yolo_object_detection(array_test_image_paths):
    num_columns = 2
    num_rows = int(np.ceil(len(array_test_image_paths) / num_columns))

    for i, test_image_path in enumerate(array_test_image_paths):
        test_image = cv.imread(test_image_path)

        # Initialize YOLOV8 class
        yolo_model(test_image)

        detected_objects = yolo_model.draw_detections(test_image)
        detected_objects = cv.cvtColor(detected_objects, cv.COLOR_BGR2RGB)

        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(detected_objects)
        plt.axis("off")

    plt.show()


test_image_path = [
    "./images/baseball.jpg",
    "./images/people.jpg",
    "./images/giraffe-zebra.jpg",
    "./images/street.jpg",
]

if __name__ == "__main__":
    yolo_object_detection(test_image_path)
