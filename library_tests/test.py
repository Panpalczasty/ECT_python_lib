import cv2
import numpy as np

imname = 'logpolar_test.png'

if __name__ == '__main__':

    img = cv2.imread(imname)

    cx = img.shape[0]/2
    cy = img.shape[1]/2

    while True:

        center = (cx, cy)
        radius = max(img.shape[:2])

        out = cv2.warpPolar(img, (0, 0), center, radius, cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG)

        inp = np.copy(img)

        inv = cv2.warpPolar(out, img.shape[:2], center, radius, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LOG)

        cv2.imshow("Result", out)

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