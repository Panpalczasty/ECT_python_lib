#!/bin/python3

import ect
import numpy as np
import cv2

img_dir = "sample_imgs/half.png"
# img_dir = "kernels/u3-v6.png"

if __name__ == "__main__":

    radius = 50

    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    log_img = ect.logpolar(img, radius, flags = ect.ECT_INTER_NONE)

    # log_img = np.expand_dims(img, 2)

    filt = ect.sidelobe(log_img.shape[:2], offset = radius/20)
    filtered = np.uint8(np.multiply(filt,log_img))
    
    ect_img = ect.ect(filtered)
    hsv = ect.complex_to_hsv(ect_img)

    # ect_img[5:, 5:] = 0
    # new_ect = np.zeros_like(ect_img)
    # h, w = ect_img.shape
    # new_ect[:h//2-10, :w//2-25] = ect_img[:h//2-10, :w//2-25]

    # test for iect
    iect_img = ect.iect(ect_img)
    # iect_img = np.fft.ifft2(ect_img)

    inv = ect.complex_to_hsv(iect_img)
    # inv = ect.norm_minmax(np.abs(iect_img), 0, 255)
    inv = ect.ilogpolar(inv) 

    cv2.imshow("Original image", img)

    cv2.imshow("Logpolar image", filtered)
    cv2.imshow("ECT of image", hsv)
    cv2.imshow("IECT of image - real", inv)

    cv2.waitKey(0)

    