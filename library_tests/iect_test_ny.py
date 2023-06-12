#!/bin/python3

import ect
import numpy as np
import cv2

img_dir = "sample_imgs/half.png"
# img_dir = "kernels/u3-v6.png"

if __name__ == "__main__":

    radius = 50
    offset = np.log(radius)/30
    mode = ect.ECT_OFFSET_ORIGIN | ect.ECT_START_NY

    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    log_img = ect.logpolar(
        img, 
        radius,
        offset = offset, 
        flags = ect.ECT_INTER_NONE | mode)

    filt = ect.sidelobe(log_img.shape[:2], 
                        offset = offset, 
                        flags = mode)
    filtered = np.uint8(np.multiply(filt,log_img))
    
    ect_img = ect.ect(filtered, offset = offset, flags = mode)
    hsv = ect.complex_to_hsv(ect_img)

    # test for iect
    iect_img = ect.iect(ect_img, offset = offset, flags = mode)
    # iect_img = np.fft.ifft2(ect_img)

    inv = ect.complex_to_hsv(iect_img)
    # inv = ect.norm_minmax(np.abs(iect_img), 0, 255)
    inv = ect.ilogpolar(inv, offset = offset, flags = mode) 

    cv2.imshow("Original image", img)

    cv2.imshow("Logpolar image", filtered)
    cv2.imshow("ECT of image", hsv)
    cv2.imshow("IECT of image - real", inv)

    cv2.waitKey(0)

    