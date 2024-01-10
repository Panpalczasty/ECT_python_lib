import numpy as np

from .utils import *
from ..filters import sigmoid

def ect(
    image: np.ndarray, 
    offset: int = None,
    flags: int = ECT_ANTIALIAS | ECT_OMIT_ORIGIN | ECT_START_NY
) -> np.ndarray:
    '''
    An O(n^4) direct implementation of ECT
    '''

    if flags & ECT_OFFSET_ORIGIN and offset is None:
        raise AttributeError("Offset is required in ECT_OFFSET_ORIGIN mode.")

    V = image.shape[0]
    U = image.shape[1]
    out = np.zeros(image.shape[:2], dtype=complex)
    kernel = np.zeros(image.shape[:2], dtype=complex)
 
    rho = np.linspace(1, U-1, U)/(U-1)*np.log(U)

    if flags & ECT_START_NY:
        phi = np.linspace(-V/4, 3*V/4-1, V)/V*2*np.pi
    else:
        phi = np.linspace(0, V-1, V)/V*2*np.pi

    rhos, phis, _ = np.meshgrid(rho, phi, 0)

    if flags & ECT_OFFSET_ORIGIN:
        xs = np.exp(rhos)*np.cos(phis) - offset
        ys = np.exp(rhos)*np.sin(phis)
    else:
        xs = np.exp(rhos)*np.cos(phis)
        ys = np.exp(rhos)*np.sin(phis)

    if flags & ECT_ANTIALIAS:
        slope = .5
        n_factor_rho = 3.5
        n_factor_phi = 3.0
        rho_aa = np.clip(1/slope*(U/np.log(U)-n_factor_rho*rhos), -100, 100)
        phi_aa = np.clip(1/slope*(V/2/np.pi-n_factor_phi*phis), -100, 100)
        rho_filter = sigmoid(rho_aa)
        phi_filter = sigmoid(phi_aa)

    for u in range(-U//2, U//2):
        # print("Progress: {}/{}".format(u+U//2, U))
        for v in range(-V//2, V//2):
            # calculate kernel
            kernel = np.exp(2*rhos-2*np.pi*(0+1j)*(xs*u/U+ys*v/V))

            # if -10 < u < 10 and -10 < v < 10:
            #     cv2.imwrite(f"kernels/u{u}-v{v}.png", norm_minmax(np.real(kernel), 0, 255))

            if flags & ECT_ANTIALIAS:
                kernel *= rho_filter*phi_filter
            # perform transform
            out[v, u] = np.multiply(image, kernel).sum().sum()

    return out

