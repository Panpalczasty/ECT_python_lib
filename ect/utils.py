import numpy as np
import cv2

def complex_to_hsv(complex_image: np.ndarray) -> cv2.Mat:
    '''
    
    '''
    if len(complex_image.shape) > 2:
        complex_image = complex_image[:,:,0]

    angle, absval = np.angle(complex_image), np.abs(complex_image)
    
    angle = np.uint8((angle+np.pi)/2/np.pi*180.)
    mag = np.uint8(absval/np.max(absval)*255.)
    value = 255*np.ones(angle.shape, dtype=np.uint8)

    hsv = np.stack((angle, value, mag), axis=2)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def norm_minmax(
        image: np.ndarray, 
        norm_min: float, 
        norm_max: float, 
        dtype: type = np.uint8):
    '''
    
    '''
    norm_base = (image - np.min(image))/(np.max(image) - np.min(image))
    return dtype((norm_base*(norm_max-norm_min)+norm_min))
