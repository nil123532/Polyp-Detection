import xml.etree.ElementTree as ET
import cv2
import json
import numpy as np
import math
import os

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve
import matplotlib.pyplot as plt

from bbps import BBPS_Scorer
from object_detector import ObjectDetector
from tracker import Tracker


def get_iou(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    if (iou > 1.0):
        print(iou)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def mask(image):
    global object_detector
    mask = np.zeros(image.shape, dtype=np.uint8)

    roi_corners = np.array([[(3,85), (45,85), (45,128), (110,128), (128,95), (128,30), (110,0),
                         (21,0), (3,30)]], dtype=np.int32)

    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


obj_detector = ObjectDetector(0.001)
tracker = Tracker(min_confidence=0.0575,
                  min_new_confidence=0.17,
                  exclusive_threshold=700,
                  match_threshold=295,
                  max_unseen=17,
                  smoothing_factor=0.35)
bbps_scorer = BBPS_Scorer(model_file='model/ResNet_Cifar10_BBPS_final.h5',
                          smoothing_factor=0.2)

TRACKED = True
MIN_IOU = 0.2
NUM_IMAGES = 800

next_bbox_id = 1

images = []
annotations = []
categories = [{"supercategory": "none", "id": 1, "name": "polyp"}]

detections = []

ground_truths = []

results_small = []
results_medium = []
results_large = []

ground_truth_count_small = 0
ground_truth_count_medium = 0
ground_truth_count_large = 0
detected_count = 0

db = [('7c', 800), ('6a', 809), ('20', 1720), ('26', 500), ('68', 831),('cvc', 612)]
#db = [('cvc', 612)]
for d in db:
    for i in range(1, d[1] + 1):
        
        print(i)

        annotation = f"/content/drive/MyDrive/WorkingFiles/Scripts/" + d[0] + "/annotations/" + str(i) + ".xml"

        if os.stat(annotation).st_size == 0:
            root = None
        else:
            tree = ET.parse(annotation)
            root = tree.getroot()

        frame = cv2.imread(f"/content/drive/MyDrive/WorkingFiles/Scripts/" + d[0] + "/images/" + str(i) + ".png")

        ground_truths_image_small = []
        ground_truths_image_medium = []
        ground_truths_image_large = []

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

                    area = (xmax - xmin) * (ymax - ymin)

                    if area < 32**2:
                        ground_truths_image_small.append([xmin, ymin, xmax, ymax, False, 1.0])
                        ground_truth_count_small += 1
                    elif 32**2 <= area < 96**2:
                        ground_truths_image_medium.append([xmin, ymin, xmax, ymax, False, 1.0])
                        ground_truth_count_medium += 1
                    else:
                        ground_truths_image_large.append([xmin, ymin, xmax, ymax, False, 1.0])
                        ground_truth_count_large += 1

        height, width, channels = frame.shape

        bboxes = obj_detector.apply_model(frame)

        image = frame[22:554, 40:663]

        image = cv2.resize(image, (128, 128))
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        clean_image = np.array(mask(image_RGB))
        clean_image = clean_image.astype("float") / 255.0
        clean_image = clean_image.reshape(1, clean_image.shape[0], clean_image.shape[1], clean_image.shape[2])

        bbps_score = bbps_scorer.predict(clean_image)

        #print("{:.3f}".format(bbps_score))

        objects = tracker.update(bboxes)

        if TRACKED:
            for j in range(len(objects)):
                x1 = objects[j].mean[0] - objects[j].mean[2]
                x2 = objects[j].mean[0] + objects[j].mean[2]
                y1 = objects[j].mean[1] - objects[j].mean[3]
                y2 = objects[j].mean[1] + objects[j].mean[3]
                confidence = objects[j].confidence

                TP = False
                for gt in ground_truths_image_small:
                    if get_iou(gt[0:4], [x1, y1, x2, y2]) >= MIN_IOU:
                        if not gt[4]:
                            results_small.append([1, confidence])
                            gt[4] = True
                            TP = True
                            detected_count += 1
                        else:
                            results_small.append([0, confidence])

                for gt in ground_truths_image_medium:
                    if get_iou(gt[0:4], [x1, y1, x2, y2]) >= MIN_IOU:
                        if not gt[4]:
                            results_medium.append([1, confidence])
                            gt[4] = True
                            TP = True
                            detected_count += 1
                        else:
                            results_medium.append([0, confidence])

                for gt in ground_truths_image_large:
                    if get_iou(gt[0:4], [x1, y1, x2, y2]) >= MIN_IOU:
                        if not gt[4]:
                            results_large.append([1, confidence])
                            gt[4] = True
                            TP = True
                            detected_count += 1
                        else:
                            results_large.append([0, confidence])

                if not TP:
                    results_small.append([0, confidence])
                    results_medium.append([0, confidence])
                    results_large.append([0, confidence])

        else:
            for j in range(len(bboxes)):
                confidence = bboxes[j, 1]

                TP = False
                for gt in ground_truths_image_small:
                    if get_iou(gt[0:4], bboxes[j, 2:6]) >= MIN_IOU:
                        if not gt[4]:
                            results_small.append([1, confidence])
                            gt[4] = True
                            TP = True
                            detected_count += 1
                        else:
                            results_small.append([0, confidence])

                for gt in ground_truths_image_medium:
                    if get_iou(gt[0:4], bboxes[j, 2:6]) >= MIN_IOU:
                        if not gt[4]:
                            results_medium.append([1, confidence])
                            gt[4] = True
                            TP = True
                            detected_count += 1
                        else:
                            results_medium.append([0, confidence])

                for gt in ground_truths_image_large:
                    if get_iou(gt[0:4], bboxes[j, 2:6]) >= MIN_IOU:
                        if not gt[4]:
                            results_large.append([1, confidence])
                            gt[4] = True
                            TP = True
                            detected_count += 1
                        else:
                            results_large.append([0, confidence])

                if not TP:
                    results_small.append([0, confidence])
                    results_medium.append([0, confidence])
                    results_large.append([0, confidence])

    tracker.reset()

#print(results)

y_true = np.asarray(results_small)[:, 0]
y_scores = np.asarray(results_small)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

#print(precision, recall)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.title("Small AP: " + str(average_precision_score(y_true, y_scores)))
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.plot(recall, precision)
plt.show()

y_true = np.asarray(results_medium)[:, 0]
y_scores = np.asarray(results_medium)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

#print(precision, recall)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.title("Medium AP: " + str(average_precision_score(y_true, y_scores)))
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.plot(recall, precision)
plt.show()

y_true = np.asarray(results_large)[:, 0]
y_scores = np.asarray(results_large)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

#print(precision, recall)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.title("Large AP: " + str(average_precision_score(y_true, y_scores)))
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.plot(recall, precision)
plt.show()

#fpr, tpr, thresholds = roc_curve(y_true, y_scores)

#fps = (fpr * (len(results) - detected_count) / NUM_IMAGES)

#plt.plot(fps, tpr)
#plt.ylabel("Recall")
#plt.xlabel("Average Number of False Positives Per Frame")
#plt.xlim(-0.1, 3)
#plt.ylim(-0.1, 1.1)
#plt.show()