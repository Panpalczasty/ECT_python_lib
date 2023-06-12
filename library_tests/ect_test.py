#!/bin/python3

import ect
import numpy as np
import cv2

img_dir = "sample_imgs/stripes1.png"

if __name__ == "__main__":

    cx = 128
    cy = 128
    radius = 50

    img = cv2.imread(img_dir)
    log_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # log_img = np.expand_dims(log_img, 2)    

    log_img = ect.logpolar(img, radius, center=(cx, cy), flags=ect.ECT_INTER_NONE)
    
    filt = ect.sidelobe(log_img.shape[:2], offset = radius/4)
    filtered = np.uint8(np.multiply(filt,log_img))
    
    # inv = maps.ilogpolar(filtered, img.shape[:2], (cx,cy), radius, maps.LOGPOLAR_INCLUDE_ORIGIN | maps.LOGPOLAR_INTER_NONE)
    
    ect_img = ect.ect(filtered)
    hsv = ect.complex_to_hsv(ect_img)
    hsv = cv2.resize(hsv, None, fx=3, fy=3)

    # ft = np.fft.fft2(img)
    # cv2.imshow("FT", ect.complex_to_hsv(ft))

    cv2.imshow("Original image", img)
    cv2.imshow("Logpolar image", filtered)
    cv2.imshow("ECT of image", hsv)

    cv2.waitKey(0)
