import cv2

from .utils import *

def ifect(
    ect: cv2.Mat,
    img_offset: int = None,
    ect_offset: int = None,
    flags: int = ECT_OFFSET_ORIGIN + ECT_ANTIALIAS + ECT_START_NY,
    aa_factors: list[float] = [.27, .15],
    aa_thresholds: list[float] = None,
    aa_slope: float = 0.25
) -> cv2.Mat:
    '''
    Implementation of Inverse FECT O(n^2)
    '''

    flags |= ECT_IECT

    if flags & ECT_OFFSET_ORIGIN and (img_offset is None or ect_offset is None):
        raise AttributeError("Offset is required in ECT_OFFSET_ORIGIN mode.")

    P, R = ect.shape[:2]
    rhos, phis, xs, ys = kernel_vectors((P, R), flags)

    kernel = np.exp(2 * np.pi * 1j * xs)
    
    if aa_thresholds is None:
        aa_thresholds = [np.log(R), 2*np.pi]

    if flags & ECT_ANTIALIAS:
        kernel = antialias(
            kernel, 
            vectors = [xs, ys],
            factors = aa_factors,
            thresholds = aa_thresholds,
            slope = aa_slope)
        
    if flags & ECT_OFFSET_ORIGIN:
        shift_ = shift(ect, ect_offset, img_offset, flags)

    image_padded = mod_image(ect, ect_offset, img_offset, flags)
    out = xcorr(image_padded, kernel)
    out = out[:P, :R]

    return shift_ * out if flags & ECT_OFFSET_ORIGIN else out