import numpy as np
import cv2
import ect

template_path = ""
sequence_path = ""


radius = 50

def get_template_spectrum(t_path: str) -> np.ndarray:
    t = cv2.imread(t_path)
    t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    t = ect.logpolar(t, radius)
    t = ect.ect(t)
    return t    


if __name__ == "__main__":

    # load and transform template
    template = get_template_spectrum(template_path)
    
    while True: # sequence

        pass
        # get new image from a sequence

        # get transform image

        # insert

        # apply phase filter



