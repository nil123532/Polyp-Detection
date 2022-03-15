import xml.etree.ElementTree as ET
import cv2
import json
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from object_detector import ObjectDetector
from tracker import Tracker

obj_detector = ObjectDetector(0.01)

next_bbox_id = 1

images = []
annotations = []
categories = [{"supercategory": "none", "id": 1, "name": "polyp"}]

bboxes = []

for i in range(1, 601):
    print(i)

    tree = ET.parse(f"/content/drive/MyDrive/WorkingFiles/Scripts/c4/annotations/{i}.xml")
    root = tree.getroot()

    frame = cv2.imread(f"/content/drive/MyDrive/WorkingFiles/Scripts/c4/images/{i}.png")

    frame2 = frame.copy()

    height, width, channels = frame.shape

    #has_annotation = False

    for child in root:
        if child.tag == "object":
            #has_annotation = True
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

            frame2 = cv2.rectangle(frame2, (xmin, ymin), (xmax, ymax), (255, 127, 0), 2)

    #if not has_annotation:
    #    continue

    image = dict()
    image["file_name"] = f"{i}.png"
    image["height"] = height
    image["width"] = width
    image["id"] = i
    images.append(image)

    bboxes.append(obj_detector.apply_model(frame))

gt_file = open("gt.json", "w+")
gt_file.write(json.dumps({'images': images, 'annotations': annotations, 'categories': categories}, indent=4))

min_confidence_space = np.linspace(0.0575, 0.06, 1)
min_new_confidence_space = np.linspace(0.17, 0.17, 1)
exclusive_threshold_space = np.linspace(700, 710, 1)
match_threshold_space = np.linspace(295, 300, 1)
max_unseen_space = np.linspace(17, 19, 1)
smoothing_factor_space = np.linspace(0.35, 0.35, 1)
area_weight_space = np.linspace(0.4, 0.6, 1)

results = np.zeros((len(min_confidence_space),
                    len(min_new_confidence_space),
                    len(exclusive_threshold_space),
                    len(match_threshold_space),
                    len(max_unseen_space),
                    len(smoothing_factor_space),
                    len(area_weight_space)))

for mc, min_confidence in enumerate(min_confidence_space):
    for mnc, min_new_confidence in enumerate(min_new_confidence_space):
        for et, exclusive_threshold in enumerate(exclusive_threshold_space):
            for mtc, match_threshold_confidence in enumerate(match_threshold_space):
                for mu, max_unseen in enumerate(max_unseen_space):
                    for sf, smoothing_factor in enumerate(smoothing_factor_space):
                        for aw, area_weight in enumerate(area_weight_space):
                            detections = []

                            tracker = Tracker(min_confidence,
                                              min_new_confidence,
                                              exclusive_threshold,
                                              match_threshold_confidence,
                                              max_unseen,
                                              smoothing_factor,
                                              area_weight)

                            for i in range(1, 601):
                                #print(i)

                                objects = tracker.update(bboxes[i - 1])

                                for j in range(len(objects)):
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

                            dt_file = open("dt.json", "w+")
                            dt_file.write(json.dumps(detections, indent=4))

                            gt = COCO("gt.json")
                            dt = gt.loadRes("dt.json")

                            coco_eval = COCOeval(gt, dt, iouType="bbox")
                            coco_eval.evaluate()
                            coco_eval.accumulate()
                            coco_eval.summarize()

                            results[mc, mnc, et, mtc, mu, sf, aw] = coco_eval.stats[1]

print(results)

print("gt:")
#print(coordinates_gt)
#print("det:")
#print(coordinates_det)