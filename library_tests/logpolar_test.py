#!/bin/python3

import ect
import numpy as np
import math
import cv2

# imname = 'logpolar_test.png'
imname = "sample_imgs/logpolar_test.png"

if __name__ == '__main__':

    img = cv2.imread(imname)

    cx = 100
    cy = 100
    radius = 50 # min(img.shape[:2])

    a = 5

    while True:
        # out = cv2.warpPolar(img, (0, 0), center, radius, cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG + cv2.INTER_NEAREST)
        out = ect.logpolar(
            img, 
            radius, 
            center=(cx, cy), 
            offset = a, 
            flags = ect.ECT_OMIT_ORIGIN | ect.ECT_START_NY
            )

        f = ect.sidelobe(out.shape[:2], offset = a, flags = ect.ECT_OMIT_ORIGIN | ect.ECT_RGB | ect.ECT_START_NY)
        out = np.uint8(np.multiply(out, f))

        # inv = cv2.warpPolar(out, img.shape[:2], center, radius, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LOG)
        inv = ect.ilogpolar(out, offset = a, flags = ect.ECT_OMIT_ORIGIN | ect.ECT_START_NY)

        # cv2.imshow("Result - cv", out)
        inp = np.copy(img)
        cv2.circle(inp, (int(cx), int(cy)), 3, (0,0,255), 2)
        cv2.imshow("Original", inp)
        cv2.imshow("Transformed", out)
        cv2.imshow("Inverted", inv)

        key = cv2.waitKey(0)

        if key == ord('w'):
            cy -= 1
        elif key == ord('s'):
            cy += 1
        elif key == ord('a'):
            cx -= 1
        elif key == ord('d'):
            cx += 1
        elif key == ord('q'):
            break 

    # cv2.imwrite("logpolar_out.png", out)
