import xml.etree.ElementTree as ET
import cv2
import json
import numpy as np
import math
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

from object_detector import ObjectDetector
from tracker import Tracker

obj_detector = ObjectDetector(0.001)
tracker = Tracker(min_confidence=0.0575,
                  min_new_confidence=0.17,
                  exclusive_threshold=700,
                  match_threshold=295,
                  max_unseen=17,
                  smoothing_factor=0.35)

TRACKED = False
SEGMENTED = False

next_bbox_id = 1

images = []
annotations = []
categories = [{"supercategory": "none", "id": 1, "name": "polyp"}]

detections = []

ground_truths = []

results = []

for i in range(1, 197):
    print(i)

    annotation = f"/home/david/PaddleDetection/dataset/etis/annotations/{i}.xml"

    if os.stat(annotation).st_size == 0:
        root = None
    else:
        tree = ET.parse(annotation)
        root = tree.getroot()

    frame = cv2.imread(f"/home/david/PaddleDetection/dataset/etis/images/{i}.tif")
    if SEGMENTED:
        segment = cv2.imread(f"/home/david/Downloads/ETIS-LaribPolypDB/Ground Truth/p{i}.tif")

    ground_truths_image = []

    if root is not None:
        for child in root:
            if child.tag == "object":
                xmin = 0
                xmax = 0
                ymin = 0
                ymax = 0

                for position in child[-1]:
                    if position.tag == "xmin":
                        xmin = int(position.text)

                    if position.tag == "xmax":
                        xmax = int(position.text)

                    if position.tag == "ymin":
                        ymin = int(position.text)

                    if position.tag == "ymax":
                        ymax = int(position.text)

                ground_truths_image.append([xmin, ymin, xmax, ymax, False, 1.0])

    height, width, channels = frame.shape

    bboxes = obj_detector.apply_model(frame)

    objects = tracker.update(bboxes)

    if TRACKED:
        for j in range(len(objects)):
            detection = dict()
            x = objects[j].mean[0]
            y = objects[j].mean[1]
            confidence = objects[j].confidence

            seg_val = 1
            if SEGMENTED:
                seg_val = np.sum(segment[math.floor(y):math.ceil(y), math.floor(x):math.ceil(x)])

            TP = False
            for gt in ground_truths_image:
                if gt[0] <= x <= gt[2] and gt[1] <= y <= gt[3]:
                    if not gt[4] and seg_val != 0:
                        results.append([1, confidence])
                        gt[4] = True
                        TP = True
                    else:
                        results.append([0, confidence])

            if not TP:
                results.append([0, confidence])

    else:
        for j in range(len(bboxes)):
            x = (bboxes[j, 2] + bboxes[j, 4]) / 2.0
            y = (bboxes[j, 3] + bboxes[j, 5]) / 2.0
            confidence = bboxes[j, 1]

            seg_val = 1
            if SEGMENTED:
                seg_val = np.sum(segment[math.floor(y):math.ceil(y), math.floor(x):math.ceil(x)])

            TP = False
            for gt in ground_truths_image:
                #
                if gt[0] <= x <= gt[2] and gt[1] <= y <= gt[3]:
                    if not gt[4] and seg_val != 0:
                        results.append([1, confidence])
                        gt[4] = True
                        TP = True
                    else:
                        results.append([0, confidence])

            if not TP:
                results.append([0, confidence])

#print(results)

y_true = np.asarray(results)[:, 0]
y_scores = np.asarray(results)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

#print(precision, recall)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.title("AP: " + str(average_precision_score(y_true, y_scores)))
plt.plot(recall, precision)
plt.show()

#last_prec = 0
#area = 0

#for i in range(len(precision)):
#    area += (precision[i] - last_prec) * recall[i]
#    last_prec = precision[i]

#print(area)