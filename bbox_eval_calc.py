from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt = COCO("gt.json")
dt = gt.loadRes("dt.json")

coco_eval = COCOeval(gt, dt, iouType="bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

#print("gt:")
#print(coordinates_gt)
#print("det:")
#print(coordinates_det)