import cv2
import numpy as np

from bbps import BBPS_Scorer

bbps_scorer = BBPS_Scorer(model_file='model/ResNet_Cifar10_BBPS_final.h5',
                          smoothing_factor=0.2)

def mask(image):
    mask = np.zeros(image.shape, dtype=np.uint8)

    roi_corners = np.array([[(3,85), (45,85), (45,128), (110,128), (128,95), (128,30), (110,0),
                         (21,0), (3,30)]], dtype=np.int32)

    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

vid = cv2.VideoCapture("video.avi")

while(vid.isOpened()):
    ret, frame = vid.read()
    if ret == True:
        image = frame[22:554, 40:663]

        image = cv2.resize(image, (128, 128))
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        clean_image = np.array(mask(image_RGB))
        clean_image = clean_image.astype("float") / 255.0
        clean_image = clean_image.reshape(1, clean_image.shape[0], clean_image.shape[1], clean_image.shape[2])

        bbps_score = bbps_scorer.predict(clean_image)

        print(bbps_score)
    else:
        break