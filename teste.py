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
#backSub = cv2.createBackgroundSubtractorMOG2()
backSub = cv2.createBackgroundSubtractorKNN(history = 10, dist2Threshold = 400.0, detectShadows = False)
ret, images = cv2.imreadmulti(
    'C:/Users/marci/Documents/projetos_code/bands fish/videos/histo.tif', [], cv2.IMREAD_GRAYSCALE)

for image in images:
    
    img_3channels = cv2.merge((image,image,image))
    
    filtered_contours = []
    
    image = cv2.GaussianBlur(image, (7, 7), 0)

    fgMask = backSub.apply(image)
    
    contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for idx, cnt in enumerate(contours):      
        area = cv2.contourArea(cnt)    
        if area > 5:
            filtered_contours.append(cnt)
    
    for c in filtered_contours:
        # compute the center of the contour
        try:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            cv2.circle(img_3channels, (cX, cY), 20, (0, 0, 255), 5)
        
        except:
           pass
    
    
    
    imS = cv2.resize(img_3channels, (1000, 500))               
    cv2.imshow("output", imS)
  
   
    imSs = cv2.resize(fgMask, (1000, 500))               
    cv2.imshow("result", imSs)
    
    if cv2.waitKey(200) & 0xFF == ord('q'):
      break