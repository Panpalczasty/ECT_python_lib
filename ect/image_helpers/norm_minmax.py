import numpy as np
import cv2

def norm_minmax(
        image: np.ndarray, 
        norm_min: float, 
        norm_max: float, 
        dtype: type = np.uint8):
    '''
    
    '''
    norm_base = (image - np.min(image))/(np.max(image) - np.min(image))
    return dtype((norm_base*(norm_max-norm_min)+norm_min))
