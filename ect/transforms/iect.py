import numpy as np
import math

from .utils import *
from ..filters import sigmoid

def iect(
    image: np.ndarray,
    offset: int = None,
    flags: int = ECT_ANTIALIAS | ECT_OMIT_ORIGIN | ECT_START_NY
) -> np.ndarray:
    '''
    An O(n^4) implementation of IECT
    '''

    if flags & ECT_OFFSET_ORIGIN and offset is None:
        raise AttributeError("Offset is required in ECT_OFFSET_ORIGIN mode.")

    image = np.expand_dims(image, 2)
    P = image.shape[0]
    R = image.shape[1]
    out = np.zeros(image.shape[:2], dtype=complex)
    kernel = np.zeros(image.shape[:2], dtype=complex)

    u = np.r_[np.linspace(0, R//2, R//2), np.linspace(-R//2, -1, abs(-R//2))]
    v = np.r_[np.linspace(0, P//2, P//2), np.linspace(-P//2, -1, abs(-P//2))]
    # u = np.linspace(-R//2, R//2, R)/R
    # v = np.linspace(-P//2, P//2, P)/P

    us, vs, _ = np.meshgrid(u, v, 0)

    if flags & ECT_ANTIALIAS:
        slope = 0.5
        n_factor_u = 4
        n_factor_v = 4
        u_filter = sigmoid(1/slope*(R-abs(n_factor_u*us)))
        v_filter = sigmoid(1/slope*(P-abs(n_factor_v*vs)))        

    for rx in range(R):
        # print("Progress: {}/{}".format(u, N))
        for px in range(P):

            rho = rx/(R-1)*np.log(R)
        
            if flags & ECT_START_NY:
                phi = px/P*2*np.pi-P*np.pi/2
            else:
                phi = px/P*2*np.pi

            if flags & ECT_OFFSET_ORIGIN:
                x = math.exp(rho)*math.cos(phi) - offset
                y = math.exp(rho)*math.sin(phi)
            else:
                x = math.exp(rho)*math.cos(phi) 
                y = math.exp(rho)*math.sin(phi)
            
            # calculate kernel
            kernel = np.exp(2*np.pi*(0+1j)*(us*x/R + vs*y/P))

            # if px == 0 or px == P//2:
            #     cv2.imwrite(f"kernels_iect/rx{rx}-px{px}.png", complex_to_hsv(np.multiply(image, kernel)))

            if flags & ECT_ANTIALIAS:
                kernel *= u_filter*v_filter
            # perform transform
            out[px, rx] = np.multiply(image, kernel).sum().sum()

    return out
