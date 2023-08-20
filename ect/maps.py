import numpy as np
import cv2 
import math

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
                x = rho_buf * cphi + cx# - offset

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
