import numpy as np
import cv2
import ect

from numpy import ndarray


def max_index(image: ndarray) -> tuple[float, float]:
    max_idx = np.argmax(image)
    max_y, max_x, _ = np.unravel_index(max_idx, image.shape)

    return max_x, max_y


def center_of_mass(image: ndarray) -> tuple[float, float]:

    image = image[:, :, 0]
    image = ect.norm_minmax(image, 0, 1, dtype=np.float64)
    image[image < 0.5] = 0
    moments = cv2.moments(image)

    c0 = moments["m00"]
    cx = moments["m10"]
    cy = moments["m01"]

    return cx/c0, cy/c0 