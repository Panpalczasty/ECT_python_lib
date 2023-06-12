#!/bin/python3

import ect
import numpy as np
import cv2

img_dir = "sample_imgs/dupa.png"
# img_dir = "kernels/u3-v6.png"

def main():
    radius = 100

    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    log_img = ect.logpolar(img, radius, flags = ect.ECT_INTER_LINEAR)

    
    filt = ect.sidelobe(log_img.shape[:2], offset = radius/10)
    filtered = np.uint8(filt * log_img)
    
    ect_img = ect.fect(filtered, flags=ect.ECT_ANTIALIAS)

    # ect_img = np.zeros_like(ect_img)
    # ect_img[:, :1] = 255
    # ect_img[:, 5] = 255

    iect_img = ect.ifect(ect_img, flags=ect.ECT_ANTIALIAS)

    hsv = ect.complex_to_hsv(ect_img)

    ect_img = ect.ilogpolar(ect_img, dtype=complex, flags=ect.ECT_INTER_NONE)
    
    inv = ect.norm_minmax(np.abs(iect_img), 0, 255)

    cv2.imshow("Original image", ect.ilogpolar(filtered))

    # cv2.imshow("Logpolar image", filtered)
    cv2.imshow("ECT of image", ect.ilogpolar(hsv))
    cv2.imshow("IECT of image", ect.ilogpolar(inv))

    cv2.waitKey(0)


if __name__ == "__main__":
    main()

    