from ect import maps, filt
import numpy as np
import math
import cv2

imname = 'spatfreq.jpg'

if __name__ == '__main__':

    img = cv2.imread(imname)
    img = cv2.resize(img, None, None, 2, 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cx = 225
    cy = 206
    radius = 100 # min(img.shape[:2])

    while True:

        center = (cx, cy)
        out = cv2.warpPolar(img, (0, 0), center, radius, cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR)
        # out = maps.logpolar(
            # img, 
            # (0,0), 
            # center, 
            # radius, 
            # maps.LOGPOLAR_INTER_LINEAR | maps.LOGPOLAR_INCLUDE_ORIGIN)

        inp = np.copy(img)

        f = filt.sidelobe(
            out.shape[:2], 
            radius,
            offset = math.log(radius/50),
            flags = filt.SIDELOBE_INCLUDE_ORIGIN | filt.SIDELOBE_GRAYSCALE)

        # apply sidelobe filter
        out = np.uint8(np.multiply(f, out))

        # prepare image for dft - normalize & grayscale
        # out = cv2.normalize(out, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_64FC1)

        out_freq = cv2.dft(out.astype(np.float64)) 
        mag_freq = np.abs(out_freq)

        mag_freq = cv2.normalize(mag_freq[1:,1:], None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_64FC1)

        # inv = cv2.warpPolar(out, img.shape[:2], center, radius, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LOG)
        inv = maps.ilogpolar(
            out, 
            img.shape[:2], 
            center, 
            radius, 
            maps.LOGPOLAR_INTER_LINEAR | maps.LOGPOLAR_INCLUDE_ORIGIN)

        cv2.imshow("Logpolar", out)
        cv2.imshow("FFT", mag_freq)
        cv2.circle(inp, (int(cx), int(cy)), 3, (0,0,255), 2)
        cv2.imshow("Original", inp)
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