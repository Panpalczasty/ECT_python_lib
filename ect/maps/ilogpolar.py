import cv2
import math
import numpy as np


from .utils import *

def ilogpolar(
    img: cv2.Mat, 
    dsize: tuple[int, int] = None,  
    center: tuple[int, int] = None,
    radius: int = None,
    offset: int = None,
    dtype = np.uint8,
    flags: int = ECT_INTER_LINEAR | ECT_OFFSET_ORIGIN | ECT_START_NY
    ) -> cv2.Mat:
    """Performs inverse logarithmic polar mapping on a source image. 

    Args:
        img (cv2.Mat): source image
        dsize (tuple[int, int]): destination image shape
        center (tuple[int, int]): center of transformed region
        radius (int): radius of transformed region
        offset (int): origin offset, required in ECT_OFFSET_ORIGIN
        flags (int, optional): execution flags. Defaults to LOGPOLAR_INTER_NONE.

    Returns:
        cv2.Mat: Inverse polar mapped source image
    """
    if flags & ECT_OFFSET_ORIGIN and offset is None:
        raise AttributeError("Offset is required in ECT_OFFSET_ORIGIN mode.")

    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]

    # get radius 
    if radius is None or radius <= 0:
        radius = img.shape[1]

    # get dsize
    if dsize is None or dsize[1] <= 0 or dsize[0] <= 0:
        out_width = round(2*radius)
        out_height = round(2*radius)
    else:
        out_width = round(dsize[1])
        out_height = round(dsize[0])

    # get center
    if center is None or center[0] <= 0 or center[1] <= 0:
        cx = cy = round(radius)
    else:
        cx = round(center[0])
        cy = round(center[1])

    out = np.zeros((out_height, out_width, 3), dtype=dtype)

    for y in range(out_height):
        for x in range(out_width):

            xc = x - cx
            yc = y - cy

            # scaling
            Kmag = img.shape[1]/math.log(radius)/2
            Kang = img.shape[0]/2/math.pi 

            # magnitude
            if flags & ECT_INCLUDE_ORIGIN:
                rho = Kmag * math.log(xc**2 + yc**2 + 1)
            elif flags & ECT_OFFSET_ORIGIN:
                xoff = xc + offset if xc > 0 else xc - offset 
                rho = Kmag * math.log(xoff**2 + yc**2 + 1e-6)
            else:
                rho = Kmag * math.log(xc**2 + yc**2 + 1e-6)

            # phase
            if flags & ECT_OFFSET_ORIGIN and flags & ECT_START_NY:
                phi = Kang * math.atan2(xoff, -yc)
            elif flags & ECT_OFFSET_ORIGIN:
                phi = Kang * math.atan2(yc, xoff)
            elif flags & ECT_START_NY:
                phi = Kang * math.atan2(xc, -yc)
            else:
                phi = Kang * math.atan2(yc, xc)


            if phi < 0:
                phi += img.shape[0]

            if flags & ECT_INTER_NONE:
                rho = round(rho)
                phi = round(phi)
                if inRange(rho, phi, img.shape[1], img.shape[0]):
                    out[y, x, :] = img[phi, rho, :]
            else:
                out[y, x, :] = bilinear_map(rho, phi, img)

    return out[:,:] if out.shape[2] == 0 else out
