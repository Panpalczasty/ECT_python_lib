import math
import cv2
import numpy as np

from .utils import *

def logpolar(
    img: cv2.Mat, 
    radius: int, 
    dsize: tuple[int, int] = None, 
    center: tuple[int, int] = None, 
    offset: int = None,
    dtype = np.uint8,
    flags: int = ECT_INTER_LINEAR | ECT_OFFSET_ORIGIN | ECT_START_NY
    ) -> cv2.Mat:
    """Performs logarithmic polar mapping on a source image. 

    Args:
        img (cv2.Mat): source image
        radius (int): radius of transformed region
        dsize (tuple[int, int]): destination image shape
        center (tuple[int, int]): center of transformed region
        offset (int): origin offset, required in ECT_OFFSET_ORIGIN
        flags (int, optional): execution flags. Defaults to ECT_INTER_LINEAR | ECT_OMIT_ORIGIN | ECT_START_PX.

    Returns:
        cv2.Mat: Polar mapped source image
    """
    if flags & ECT_OFFSET_ORIGIN and offset is None:
        raise AttributeError("Offset is required in ECT_OFFSET_ORIGIN mode.")

    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]

    if dsize is None or dsize[1] <= 0 or dsize[0] <= 0:
        out_width = round(radius) 
        out_height = round(radius*np.pi)
    else:
        out_height, out_width = (round(x) for x in dsize)

    if center is None or not inRange(center[0], center[1], img.shape[1], img.shape[0]):
        cx = img.shape[1]//2
        cy = img.shape[0]//2
    else:    
        cx, cy = center

    out = np.zeros((out_height, out_width, img.shape[2]), dtype=dtype)

    Kmag = math.log(radius)/out_width
    Kang = 2*math.pi/out_height

    for phi in range(out_height):
        for rho in range(out_width):

            if flags & ECT_INCLUDE_ORIGIN:
                rho_buf = math.exp(Kmag*rho) - 1.0                    
            else:
                rho_buf = math.exp(Kmag*rho)

            if flags & ECT_START_NY:
                cphi = math.sin(Kang*phi)
                sphi = -math.cos(Kang*phi)
            else:
                cphi = math.cos(Kang*phi)
                sphi = math.sin(Kang*phi)

            if flags & ECT_OFFSET_ORIGIN:
                x = rho_buf * cphi + cx # - offset

                if x > cx:
                    x -= offset
                else:
                    x += offset

                y = rho_buf * sphi + cy
            else:
                x = rho_buf * cphi + cx
                y = rho_buf * sphi + cy

            if flags & ECT_INTER_NONE:
                x = round(x)
                y = round(y)
                if inRange(x, y, img.shape[1], img.shape[0]):
                    out[phi, rho, :] = img[y, x, :]
            else:
                out[phi, rho, :] = bilinear_map(x, y, img)
            
            
    return out[:,:] if out.shape[2] == 0 else out