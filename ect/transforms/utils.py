import numpy as np

from ..filters import sigmoid

ECT_NONE = 8
ECT_ANTIALIAS = 16

# origin mode flags
ECT_INCLUDE_ORIGIN = 1 
ECT_OMIT_ORIGIN = 2
ECT_OFFSET_ORIGIN = 4

# angle flags
ECT_START_PX = 32
ECT_START_NY = 64

# direction flags
ECT_ECT = 128
ECT_IECT = 256

def xcorr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''
    Calculates 2D cross-correlation between two ndarrays
    using following equation:

    xcorr(a, b) = F(A)' * F(B)

    where F stands for Fourier transform
    
    Parameters
    ----------
    A, B : np.ndarray
        Input arrays

    Returns
    -------
    np.ndarray
        Cross-correlation between two arrays
    '''
    A_t = np.fft.fft2(A, axes=(0, 1))
    B_t = np.fft.fft2(B, axes=(0, 1))
    out_t = np.conjugate(A_t) * B_t
    return np.fft.ifft2(out_t, axes=(0,1))


def kernel_vectors(
    shape: tuple[int, int], 
    flags: int = ECT_START_NY
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Generates base vectors
    for a given shape, in range 
    [1, log(R)] x (0, 2*pi)


    Parameters
    ----------
    shape : tuple[int, int]
        shape of kernel image
    flags : int, optional
        launch configuration, by default ECT_START_PX

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple of gamma, phi, x and y vectors
    '''

    P, R = shape

    gamma = np.linspace(-1+1/R, 1, 2*R) * np.log(R)
    phi = np.linspace(1/P, 2, 2*P) * 2 * np.pi

    if flags & ECT_START_NY:
        phi -= 0.5 * np.pi

    gammas, phis, _ = np.meshgrid(gamma, phi, 0)

    xs = np.exp(gammas) * np.cos(phis)
    ys = np.exp(gammas) * np.sin(phis)

    return gammas, phis, xs, ys


def image_vectors(shape: tuple[int, int], img_offset: int, flags: int = ECT_START_NY):
    '''
    Generates base vectors
    for a given shape, in range 
    [1, log(R)] x (0, 2*pi)


    Parameters
    ----------
    shape : tuple[int, int]
        shape of kernel image
    flags : int, optional
        launch configuration, by default ECT_START_PX

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple of gamma, phi, x and y vectors
    '''

    P, R = shape

    rho = np.linspace(1/R, 1, R) * np.log(R)
    phi = np.linspace(1/P, 1, P) * 2 * np.pi

    if flags & ECT_START_NY:
        phi -= 0.5 * np.pi

    rhos, phis, _ = np.meshgrid(rho, phi, 0)

    xs = np.exp(rhos) * np.cos(phis)
    ys = np.exp(rhos) * np.sin(phis)

    # if flags & ECT_OFFSET_ORIGIN:
        # xs[:P//2, :] -= img_offset
        # xs[P//2:, :] += img_offset

    return rhos, phis, xs, ys


def antialias(
    kernel: np.ndarray, 
    vectors: list[np.ndarray],
    factors: list[float],
    thresholds: list[float],
    slope: float):
    '''
    Applies antialiasing filter to a kernel

    Parameters
    ----------
    kernel : np.ndarray
        _description_
    vectors : list[np.ndarray]
        _description_
    factors : list[np.ndarray]
        _description_
    thresholds : list[float]
        _description_
    slope : float
        _description_

    Returns
    -------
    _type_
        _description_
    '''
    for v, fact, thr in zip(vectors, factors, thresholds):  
        filt = sigmoid(1/slope*(thr-fact*abs(v)))
        kernel *= filt

    return kernel


def mod_image(
    image: np.ndarray, 
    img_offset: int,
    ect_offset: int, 
    flags: int = ECT_OMIT_ORIGIN | ECT_START_NY | ECT_ECT):
    '''
    Prepares imadd

    Parameters
    ----------
    image : np.ndarray
        _description_
    ect_offset : int
        _description_
    flags : int, optional
        _description_, by default ECT_OMIT_ORIGIN | ECT_START_PX

    Returns
    -------
    _type_
        _description_
    '''
    P, R = image.shape[:2]
    image_padded = np.zeros((2*P, 2*R, 1), dtype=complex)

    rhos, _, xs, _ = image_vectors((P, R), img_offset, flags)

    ect_factor = 1 if flags & ECT_ECT else -1

    if flags & ECT_OFFSET_ORIGIN:
        image_padded[:P, :R] = np.conjugate(image) * \
            np.exp(2*rhos - ect_factor*2*np.pi*1j*ect_offset*xs/R)
    else:
        image_padded[:P, :R] = np.conjugate(image) * np.exp(2*rhos)

    return image_padded


def shift(
    image: np.ndarray, 
    img_offset: int, 
    ect_offset: int, 
    flags: int = ECT_START_NY | ECT_ECT):

    P, R = image.shape[:2]
    _, _, xs, _ = image_vectors((P, R), ect_offset, flags)

    ect_factor = 1 if flags & ECT_ECT else -1

    return np.exp(ect_factor*2*np.pi*1j*img_offset*xs/R)
