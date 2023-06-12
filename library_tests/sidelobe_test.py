import ect
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

img_dir = "logpolar_test.png"

if __name__ == "__main__":

    cx = 225
    cy = 206
    radius = 50

    img = cv2.imread(img_dir)

    log_img = ect.logpolar(img, radius, center=(cx, cy))
    
    rho = np.linspace(1, math.log(radius), log_img.shape[1])
    phi = np.linspace(0, 2*math.pi, log_img.shape[0])
    rho, phi = np.meshgrid(rho, phi)
    
    filt = ect.sidelobe(
        log_img.shape[:2],
        offset = np.log(radius/10),
        flags = ect.ECT_RGB | ect.ECT_INCLUDE_ORIGIN)

    filtered = np.uint8(np.multiply(filt,log_img))
    
    inv = ect.ilogpolar(filtered)

    print(np.min(filt), np.max(filt))

    cv2.imshow("Sidelobe filter", filt)
    cv2.imshow("Sidelobe filtered image", filtered)
    cv2.imshow("Sidelobe filter inv", inv)
    cv2.waitKey(0)

    