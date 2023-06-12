#!/bin/python3

import ect
import numpy as np
import cv2

img_dir = "sample_imgs/text.png"
# img_dir = "kernels/u3-v6.png"

def main():
    radius = 100
    mode = ect.ECT_START_PX | ect.ECT_OMIT_ORIGIN

    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    log_img = ect.logpolar(img, radius, offset= radius/10, flags = ect.ECT_INTER_LINEAR | mode)
    
    filt = ect.sidelobe(log_img.shape[:2], offset = radius/10, flags = mode)
    filtered = np.uint8(filt * log_img)
    # filtered = log_img

    ect_img = ect.fect(filtered, offset = radius/10, ect_offset=0, flags= ect.ECT_ANTIALIAS | mode)

    # ect_img = np.zeros_like(ect_img)
    # ect_img[:, :3] = 255
    # ect_img[50:60, 90:100] = 255

    iect_img = ect.ifect(ect_img, offset = 0, ect_offset=radius/10, flags = ect.ECT_ANTIALIAS | mode)

    hsv = ect.complex_to_hsv(ect_img)

    ect_img = ect.ilogpolar(ect_img, dtype=complex, offset = radius/10, flags=ect.ECT_INTER_NONE | mode)
    
    inv = ect.norm_minmax(np.abs(iect_img), 0, 255)
    # inv = ect.complex_to_hsv(iect_img)

    cv2.imshow("Original image", filtered)

    # cv2.imshow("Logpolar image", filtered)
    cv2.imshow("ECT of image", ect.ilogpolar(hsv, offset=radius/10, flags=mode))
    cv2.imshow("IECT of image", ect.ilogpolar(inv, offset=radius/10, flags=mode))

    cv2.waitKey(0)


if __name__ == "__main__":
    main()

    