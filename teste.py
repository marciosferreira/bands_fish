#ret, image = cv2.imreadmulti('LanB2_KD1P2_31603_gut_2.tif', [], cv2.IMREAD_ANYCOLOR)
# You can then access each layer by their indexes (eg. image[0]).

from matplotlib import pyplot as plt
import time
import pandas as pd
import math
import numpy as np
import cv2
from skimage import io

import scipy.stats
backSub = cv2.createBackgroundSubtractorMOG2(
    history=10, varThreshold=40, detectShadows=False)
#backSub = cv2.createBackgroundSubtractorKNN(history = 10, dist2Threshold = 800.0, detectShadows = False)
ret, images = cv2.imreadmulti(
    'C:/Users/marci/Documents/projetos_code/bands fish/videos/histo.tif', [], cv2.IMREAD_GRAYSCALE)


fish_1 = None
fish_2 = None
fish_3 = None
fish_4 = None
fish_5 = None
fish_6 = None
fish_7 = None
fish_8 = None
fish_9 = None
fish_10 = None
fish_11 = None
fish_12 = None
fish_13 = None
fish_14 = None
fish_15 = None

for image in images:

    img_3channels = cv2.merge((image, image, image))

    filtered_contours = []

    image = cv2.GaussianBlur(image, (7, 7), 0)

    fgMask = backSub.apply(image)

    contours, hierarchy = cv2.findContours(
        fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 5:
            filtered_contours.append(cnt)

    for c in filtered_contours:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        if 0 < cY < 138:
            fish_1 = (cX, cY)
        if 138 < cY < 261:
            fish_2 = (cX, cY)
        if 261 < cY < 390:
            fish_3 = (cX, cY)
        if 390 < cY < 519:
            fish_4 = (cX, cY)
        if 519 < cY < 636:
            fish_5 = (cX, cY)
        if 636 < cY < 762:
            fish_6 = (cX, cY)
        if 762 < cY < 897:
            fish_7 = (cX, cY)
        if 897 < cY < 1008:
            fish_8 = (cX, cY)
        if 1008 < cY < 1140:
            fish_9 = (cX, cY)
        if 1140 < cY < 1263:
            fish_10 = (cX, cY)
        if 1263 < cY < 1383:
            fish_11 = (cX, cY)
        if 1383 < cY < 1515:
            fish_12 = (cX, cY)
        if 1515 < cY < 1638:
            fish_13 = (cX, cY)
        if 1638 < cY < 1761:
            fish_14 = (cX, cY)
        if 1761 < cY < 1896:
            fish_15 = (cX, cY)

    list_values = [fish_1, fish_2, fish_3, fish_4, fish_5, fish_6, fish_7,
                   fish_8, fish_9, fish_10, fish_11, fish_12, fish_13, fish_14, fish_15]
    # draw the contour and center of the shape on the image

    for i in list_values:
        if i is not None:
            cv2.circle(img_3channels, (i[0], i[1]), 20, (0, 0, 255), 5)

    imS = cv2.resize(img_3channels, (1000, 500))
    cv2.imshow("output", imS)

    imSs = cv2.resize(fgMask, (1000, 500))
    cv2.imshow("result", imSs)

    if cv2.waitKey(200) & 0xFF == ord('q'):
        break
