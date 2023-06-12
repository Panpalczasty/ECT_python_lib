#!/bin/python3

import ect
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

if __name__ == "__main__":

    radius = 50

    img = 255*np.ones((100,100,1), dtype=np.uint8)
    log_img = ect.logpolar(img, radius)    

    # examine single ECT points
    ect_img = np.zeros(log_img.shape, dtype=complex)
    ect_img[0,0,:] = 1
    ect_img[1,0,:] = 1
    # ect_img[0,2] = 1.
    # ect_img[-1,-1] = 1.
    # ect_img[-1,-11] = 1.


    # test for iect
    iect_img = ect.ect(ect_img, flags= ect.ECT_NONE)

    inv = ect.norm_minmax(np.abs(iect_img), 0, 255)
    # inv = ect.ilogpolar(inv) 
    # inv = np.uint8(inv*255)

    cv2.imshow("Original image", img[:100,:100])

    cv2.imshow("ECT of image", ect.complex_to_hsv(ect_img))

    cv2.imshow("Inverted image", inv)

    cv2.waitKey(0)

    