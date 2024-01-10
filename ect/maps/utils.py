import math
import numpy as np

# padding flag
ECT_FILL_OUTLIERS = 128

# origin mode flags
ECT_INCLUDE_ORIGIN = 1 
ECT_OMIT_ORIGIN = 2
ECT_OFFSET_ORIGIN = 4

# interpolation flags
ECT_INTER_LINEAR = 8
ECT_INTER_NONE = 16

# angle flags
ECT_START_PX = 32 # start from positive x
ECT_START_NY = 64 # start from negative y (as in src paper)

def inRange(x: int, y: int, width: int, height: int) -> bool:
    """Checks if a point (x,y) is in range
    from (0,0) to (width, height) i.e. x and y are
    valid indices for an array of shape width x height
    
    Args:
        x (int): horizontal coordinate/index
        y (int): vertical coordinate/index
        width (int): width of array/picture
        height (int): height of array/picture

    Returns:
        bool: True if (x, y) is a valid index, otherwise False
    """

    inx = x >= 0 and x < width
    iny = y >= 0 and y < height

    return inx and iny 


def get_int_bound(x: float) -> tuple[int, int]:
    """Returns lower and upper integer boundaries of x,
    satisfying equation: x_l <= x < x_u
    
    Args:
        x (float): A number to be bounded

    Returns:
        tuple[int, int]: lower and upper boundary of a number
    """

    if x.is_integer():
        return int(x), int(x+1)
    else:
        return math.floor(x), math.ceil(x)


def bilinear_map(x: float, y: float, img: cv2.Mat) -> np.uint8:
    """Performs bilinear interpolation on nearest neighbors
    of a point (x,y) on an image and returns rounded pixel value

    Args:
        x (float): horizontal coordinate
        y (float): vertical coordinate
        img (cv2.Mat): source image

    Returns:
        np.uint8: interpolated pixel value
    """

    xlow, xhi = get_int_bound(x)
    ylow, yhi = get_int_bound(y)

    if inRange(xlow, ylow, img.shape[1]-1, img.shape[0]-1):
        px1 = img[ylow, xlow, :]
        px2 = img[ylow, xhi, :]
        px3 = img[yhi, xlow, :]
        px4 = img[yhi, xhi, :]

        px12 = (x-xlow)*px2 + (xhi-x)*px1
        px34 = (x-xlow)*px4 + (xhi-x)*px3

        return np.uint8((y-ylow)*px34 + (yhi-y)*px12)

    else: 
        return 0

