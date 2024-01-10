import numpy as np

# origin mode flags
ECT_INCLUDE_ORIGIN = 1 
ECT_OMIT_ORIGIN = 2
ECT_OFFSET_ORIGIN = 4

# interpolation flags
ECT_RGB = 8
ECT_GRAYSCALE = 16

# angle flags
ECT_START_PX = 32
ECT_START_NY = 64

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Calculates sigmoid function of an numpy array

    Args:
        x (float): input

    Returns:
        float: output
    """
    x = np.clip(x, -100, 100)
    return 1/(1 + np.exp(-x))


def vector_gen(shape: tuple[int, int]):
    '''
    Generates rho vector.

    Parameters
    ----------
    shape : tuple[int, int]
        shape of kernel image

    Returns
    -------
    np.ndarray
        Rho vector
    '''

    P, R = shape

    rho = np.linspace(1, R, R)
    phi = np.linspace(0, 1, P)

    rhos, phis, _ = np.meshgrid(rho, phi, 0)

    return rhos, phis