
import scipy.stats
from skimage import io
import cv2
import numpy as np
import math
import pandas as pd
import time
from matplotlib import pyplot as plt
list_grid = [0, 138, 261, 390, 519, 636, 762, 897,
             1008, 1140, 1263, 1383, 1515, 1638, 1761, 1900]


backSub = cv2.createBackgroundSubtractorMOG2(
    history=50, varThreshold=20, detectShadows=False)
#backSub = cv2.createBackgroundSubtractorKNN(history = 10, dist2Threshold = 800.0, detectShadows = False)
ret, images = cv2.imreadmulti(
    'C:/Users/marci/Documents/projetos_code/bands fish/videos/histo.tif', [], cv2.IMREAD_GRAYSCALE)


#fish_1 = fish_2 = fish_3 = fish_4 = fish_5 = fish_6 = fish_7 = fish_8 = fish_9 = fish_10 = fish_11 = fish_12 = fish_13 = fish_14 = fish_15 = None

fish = {"fish_1": None, "fish_2": None, "fish_3": None, "fish_4": None, "fish_5": None, "fish_6": None, "fish_7": None,
        "fish_8": None, "fish_9": None, "fish_10": None, "fish_11": None, "fish_12": None, "fish_13": None, "fish_14": None, "fish_15": None}

video_final = []

for idx, image in enumerate(images):

    fgMask = backSub.apply(image)

    if idx > 4:

        provisional_fish = {"fish_1": None, "fish_2": None, "fish_3": None, "fish_4": None, "fish_5": None, "fish_6": None, "fish_7": None,
                            "fish_8": None, "fish_9": None, "fish_10": None, "fish_11": None, "fish_12": None, "fish_13": None, "fish_14": None, "fish_15": None}

        img_3channels = cv2.merge((image, image, image))

        filtered_contours = []

        image = cv2.GaussianBlur(image, (7, 7), 0)

        contours, hierarchy = cv2.findContours(
            fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 10:
                filtered_contours.append(cnt)

        for c in filtered_contours:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            for idx, z in enumerate(list_grid):
                if idx < len(list_grid) - 1:
                    if list_grid[idx] < cY < list_grid[idx + 1]:
                        if provisional_fish['fish_' + str(idx + 1)] is None:
                            provisional_fish.update(
                                {'fish_' + str(idx + 1): [(cX, cY)]})
                        else:
                            provisional_fish['fish_' +
                                             str(idx + 1)].append((cX, cY))

        for key, value in provisional_fish.items():
            if value is not None:
                if len(value) > 1 and fish[key] is not None:
                    results_distance = []

                    for coords in value:
                        cX_diff = abs(coords[0] - fish[key][0])
                        results_distance.append(cX_diff)

                    min_value = min(results_distance)
                    min_index = results_distance.index(min_value)
                    fish.update({key: value[min_index]})
                else:
                    fish.update({key: value[0]})

        for item in fish.items():
            if item[1] is not None:
                cv2.circle(img_3channels,
                           (item[1][0], item[1][1]), 20, (0, 0, 255), 5)

        img_w = img_3channels[0].shape[0]
        for coord in list_grid:
            start_point = (0, coord)
            end_point = (img_w, coord)
            img_3channels = cv2.line(
                img_3channels, start_point, end_point, (0, 255, 0), 4)

        imS = cv2.resize(img_3channels, (1000, 500))
        cv2.imshow("output", imS)

        imSs = cv2.resize(fgMask, (1000, 500))
        cv2.imshow("result", imSs)

        video_final.append(imS)

        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
height, width, layers = video_final[0].shape
size = (width, height)
out_vid = "C:/Users/marci/Documents/projetos_code/bands fish/videos/output.mp4"
out = cv2.VideoWriter(out_vid, fourcc, 10, size)
for i in video_final:
    out.write(i)
out.release()
