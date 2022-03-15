import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bbps import BBPS_Scorer
from cecum import Cecum_Detector
from object_detector import ObjectDetector
from tracker import Tracker

obj_detector = ObjectDetector(0.01)
tracker = Tracker(min_confidence=0.0575,
                  min_new_confidence=0.17,
                  exclusive_threshold=700,
                  match_threshold=295,
                  max_unseen=17,
                  smoothing_factor=0.35)
bbps_scorer = BBPS_Scorer(model_file='model/ResNet_Cifar10_BBPS_final.h5',
                          smoothing_factor=0.2)
cecum_detector = Cecum_Detector(model_file='model/ResNet_Cifar10_cecum.h5',
                                smoothing_factor=0.2,
                                threshold=0.5,
                                cecum_reached_count_threshold=25)

SCALE = 128

def mask(image):
    mask = np.zeros(image.shape, dtype=np.uint8)

    roi_corners = np.array([[(3,85), (45,85), (45,128), (110,128), (128,95), (128,30), (110,0),
                         (21,0), (3,30)]], dtype=np.int32)

    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

for i in range(0, 1720):
    print(i)

    frame_draw = Image.open(f"/home/david/Downloads/6ab/2033ada3-d76b-40f7-9b6b-2c09d6934f1c/{i}.png")

    frame_predict = frame_draw.copy()

    frame_draw = add_margin(frame_draw, 0, 0, 0, 250, (0, 0, 0))
    draw = ImageDraw.Draw(frame_draw)
    font = ImageFont.truetype("FreeMono.ttf", 35)

    bboxes = obj_detector.apply_model(frame_predict)

    objects = tracker.update(bboxes)

    frame_predict = frame_predict.crop((40, 22, 663, 554))

    image = np.array(frame_predict)
    image = image[:, :, ::-1].copy()
    image = cv2.resize(image, (SCALE, SCALE))
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    clean_image = np.array(mask(image_RGB))
    clean_image = clean_image.astype("float") / 255.0
    clean_image = clean_image.reshape(1, clean_image.shape[0], clean_image.shape[1], clean_image.shape[2])

    # Bowel Prep
    bbps_score = bbps_scorer.predict(clean_image)
    if bbps_score <= 0.5:
        # print('bbps', bbps_score)
        bbps_colour = (160, 27, 16)
        bbps_text = 'BBPS 0-1'
    else:
        bbps_colour = (58, 153, 68)
        bbps_text = 'BBPS 2-3'

    draw.text((5, 0), bbps_text, font=font, fill=bbps_colour)

    #draw.text((5, 100), str(bbps_score), font=font, fill=bbps_colour)

    # Cecum
    cecum_reached = cecum_detector.predict(clean_image)
    if cecum_reached:
        draw.text((5, 50), 'Cecum Reached', font=font, fill=(174, 204, 242))

    for j in range(len(objects)):
        if (bbps_score > 0.9 and objects[j].confidence > 0.33) or objects[j].confidence > 0.9:
            colour = (255, 0, 0)

            if objects[j].confidence < 0.75:
                colour = (255, 255, 0)

            if objects[j].confidence < 0.40:
                colour = (0, 255, 0)

            draw.rectangle([objects[j].mean[0] - objects[j].mean[2] + 250,
                            objects[j].mean[1] - objects[j].mean[3],
                            objects[j].mean[0] + objects[j].mean[2] + 250,
                            objects[j].mean[1] + objects[j].mean[3]],
                           fill=None, outline=colour,
                           width=3)

            # draw.text((objects[i].mean[0] - objects[i].mean[2], objects[i].mean[1] - objects[i].mean[3]),
            #          "{:.2f}".format(objects[i].confidence), colour, font=font)

    frame_draw.save(f"out/{i}.jpg")