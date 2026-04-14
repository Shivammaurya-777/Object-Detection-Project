import cv2
import numpy as np

# Load class names
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def predict(image):
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    final_boxes = []
    labels = []
    scores = []

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]

            final_boxes.append([x, y, x + w, y + h])
            labels.append(classes[class_ids[i]])
            scores.append(confidences[i])

    return final_boxes, labels, scores