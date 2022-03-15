import xml.etree.ElementTree as ET
import cv2
import json
import numpy as np
import os

from bbps import BBPS_Scorer
from object_detector import ObjectDetector
from tracker import Tracker


def mask(image):
    mask = np.zeros(image.shape, dtype=np.uint8)

    roi_corners = np.array([[(3,85), (45,85), (45,128), (110,128), (128,95), (128,30), (110,0),
                         (21,0), (3,30)]], dtype=np.int32)

    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


obj_detector = ObjectDetector(0.01)
tracker = Tracker(min_confidence=0.0575,
                  min_new_confidence=0.17,
                  exclusive_threshold=700,
                  match_threshold=295,
                  max_unseen=17,
                  smoothing_factor=0.35)
bbps_scorer = BBPS_Scorer(model_file='model/ResNet_Cifar10_BBPS_final.h5',
                          smoothing_factor=0.2)

next_bbox_id = 1

images = []
annotations = []
categories = [{"supercategory": "none", "id": 1, "name": "polyp"}]

detections = []

for i in range(1, 1721):
    print(i)

    if os.stat(f"/content/drive/MyDrive/WorkingFiles/Scripts/20/annotations/{i}.xml").st_size == 0:
        root = None
    else:
        tree = ET.parse(f"/content/drive/MyDrive/WorkingFiles/Scripts/20/annotations/{i}.xml")
        root = tree.getroot()

    frame = cv2.imread(f"/content/drive/MyDrive/WorkingFiles/Scripts/20/images/{i}.png")

    frame2 = frame.copy()

    height, width, channels = frame.shape

    image = dict()
    image["file_name"] = f"{i}.png"
    image["height"] = height
    image["width"] = width
    image["id"] = i
    images.append(image)

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

                annotation = dict()
                annotation["area"] = (xmax - (xmin - 1)) * (ymax - (ymin - 1))
                annotation["iscrowd"] = 0
                annotation["bbox"] = [xmin - 1, ymin - 1, xmax - (xmin - 1), ymax - (ymin - 1)]
                annotation["category_id"] = 1
                annotation["ignore"] = 0
                annotation["segmentation"] = []
                annotation["image_id"] = i
                annotation["id"] = next_bbox_id
                annotations.append(annotation)

                next_bbox_id += 1

                bboxes = obj_detector.apply_model(frame)

                image = frame[22:554, 40:663]

                image = cv2.resize(image, (128, 128))
                image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                clean_image = np.array(mask(image_RGB))
                clean_image = clean_image.astype("float") / 255.0
                clean_image = clean_image.reshape(1, clean_image.shape[0], clean_image.shape[1], clean_image.shape[2])

                bbps_score = bbps_scorer.predict(clean_image)

                objects = tracker.update(bboxes)

                for j in range(len(objects)):
                    #if bbps_score > 0.99:
                    detection = dict()
                    detection["bbox"] = [objects[j].mean[0] - objects[j].mean[2], objects[j].mean[1] - objects[j].mean[3],
                                         2 * objects[j].mean[2], 2 * objects[j].mean[3]]
                    detection["category_id"] = 1
                    detection["image_id"] = i
                    detection["score"] = float(objects[j].confidence)
                    detections.append(detection)

                    #if objects[j].confidence > 0.05913:
                    #    frame2 = cv2.rectangle(frame2,
                    #                           (int(objects[j].mean[0] - objects[j].mean[2]), int(objects[j].mean[1] - objects[j].mean[3])),
                    #                           (int(objects[j].mean[0] + objects[j].mean[2]), int(objects[j].mean[1] + objects[j].mean[3])),
                    #                           (255, 0, 127), 2)

                #for j in range(len(bboxes)):
                #    detection = dict()
                #    detection["bbox"] = [float(bboxes[j, 2]), float(bboxes[j, 3]),
                #                         float(bboxes[j, 4] - bboxes[j, 2]), float(bboxes[j, 5] - bboxes[j, 3])]
                #    detection["category_id"] = 1
                #    detection["image_id"] = i
                #    detection["score"] = float(bboxes[j, 1])
                #    detections.append(detection)

                #    if bboxes[j, 1] > 0.01:
                #        frame2 = cv2.rectangle(frame2,
                #                               (int(bboxes[j, 2]), int(bboxes[j, 3])),
                #                               (int(bboxes[j, 4]), int(bboxes[j, 5])),
                #                               (255, 0, 127), 2)

                #cv2.imwrite(f"out/{i}.jpg", frame2)

gt_file = open("gt.json", "w+")
gt_file.write(json.dumps({'images': images, 'annotations': annotations, 'categories': categories}, indent=4))

dt_file = open("dt.json", "w+")
dt_file.write(json.dumps(detections, indent=4))

#print("gt:")
#print(coordinates_gt)
#print("det:")
#print(coordinates_det)