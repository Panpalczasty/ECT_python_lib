import numpy as np
import cv2
import math
from ect.filt import sigmoid
from ect.utils import complex_to_hsv
from ect.utils import norm_minmax

ECT_NONE = 8
ECT_ANTIALIAS = 16

# origin mode flags
ECT_INCLUDE_ORIGIN = 1 
ECT_OMIT_ORIGIN = 2
ECT_OFFSET_ORIGIN = 4

# angle flags
ECT_START_PX = 32
ECT_START_NY = 64

def ect(
    image: np.ndarray, 
    offset: int = None,
    flags: int = ECT_ANTIALIAS | ECT_OMIT_ORIGIN | ECT_START_PX
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


def iect(
    image: np.ndarray,
    offset: int = None,
    flags: int = ECT_ANTIALIAS | ECT_OMIT_ORIGIN | ECT_START_PX
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


def xcorr(A : np.ndarray, B : np.ndarray):

    A_t = np.fft.fft2(A)
    B_t = np.fft.fft2(B)
    out_t = np.conjugate(A_t) * B_t
    return np.fft.ifft2(out_t)


def make_kernel_vectors(shape: tuple[int, int], flags: int = ECT_START_PX):

    P, R = shape

    gamma = np.linspace(-1+1/R, 1, 2*R) * np.log(R)
    phi = np.linspace(1/P, 1, P) * 2 * np.pi

    if flags & ECT_START_NY:
        phi -= 0.5 * np.pi

    gammas, phis, _ = np.meshgrid(gamma, phi, 0)

    xs = np.exp(gammas) * np.cos(phis)
    ys = np.exp(gammas) * np.sin(phis)

    return gammas, phis, xs, ys


def make_image_vectors(shape: tuple[int, int], flags: int = ECT_START_PX):

    P, R = shape

    rho = np.linspace(1/R, 1, R) * np.log(R)
    phi = np.linspace(1/P, 1, P) * 2 * np.pi

    if flags & ECT_START_NY:
        phi -= 0.5 * np.pi

    rhos, phis, _ = np.meshgrid(rho, phi, 0)

    xs = np.exp(rhos) * np.cos(phis)
    ys = np.exp(rhos) * np.sin(phis)

    return rhos, phis, xs, ys

def antialias(
    kernel: np.ndarray, 
    x_vector: np.ndarray, 
    y_vector: np.ndarray,
    x_factor: float,
    y_factor: float,
    x_threshold: float,
    y_threshold: float,
    slope: float):

        x_filter = sigmoid(1/slope*(x_threshold - x_factor*abs(x_vector)))
        y_filter = sigmoid(1/slope*(y_threshold - y_factor*abs(y_vector)))      
        return kernel * x_filter * y_filter


def mod_image(image: np.ndarray, ect_offset: int, flags: int = ECT_OMIT_ORIGIN | ECT_START_PX):

    P, R = image.shape[:2]
    image_padded = np.zeros((P, 2*R, 1), dtype=complex)

    rhos, _, xs, _ = make_image_vectors((P, R), flags)

    if flags & ECT_OFFSET_ORIGIN:
        image_padded[:P, :R] = np.conjugate(image) * np.exp(2*rhos - 2*np.pi*1j*ect_offset*xs)
    else:
        image_padded[:P, :R] = np.conjugate(image) * np.exp(2*rhos)

    return image_padded


def make_shift(image: np.ndarray, offset: int, ect_offset: int, flags: int = ECT_START_PX):

    P, R = image.shape[:2]
    _, _, xs, _ = make_image_vectors((P, R), flags)

    return np.exp(2*np.pi*1j*offset*(xs - ect_offset))

def fect(
    image: cv2.Mat,
    offset: int = None,
    ect_offset: int = None,
    flags: int = ECT_OMIT_ORIGIN & ECT_NONE & ECT_START_PX
) -> cv2.Mat:
    '''
    Implementation of Fast ECT O(n^2*logn)
    '''

    if flags & ECT_OFFSET_ORIGIN and (offset is None or ect_offset is None):
        raise AttributeError("Offset is required in ECT_OFFSET_ORIGIN mode.")

    P, R = image.shape[:2]

    _, _, xs, ys = make_kernel_vectors((P,R), flags)

    kernel = np.exp(-2 * np.pi * 1j * xs)

    if flags & ECT_ANTIALIAS:
        kernel = antialias(kernel, xs, ys, 0.15, 0, np.log(R), np.log(R), 0.1)

    if flags & ECT_OFFSET_ORIGIN:
        shift = make_shift(image, offset, ect_offset, flags)

    image_padded = mod_image(image, ect_offset)
    out = xcorr(image_padded, kernel)

    out = out[:, :R][::-1, :]

    return shift * out if flags & ECT_OFFSET_ORIGIN else out


def ifect(
    image: cv2.Mat,
    offset: int = None,
    ect_offset: int = None,
    padding_scale: int = 2,
    flags: int = ECT_OMIT_ORIGIN & ECT_NONE & ECT_START_PX
) -> cv2.Mat:
    '''
    Implementation of Inverse FECT O(n^2)
    '''

    if flags & ECT_OFFSET_ORIGIN and (offset is None or ect_offset is None):
        raise AttributeError("Offset is required in ECT_OFFSET_ORIGIN mode.")

    P, R = image.shape[:2]
    _, _, xs, ys = make_kernel_vectors((P, R), flags)

    kernel = np.exp(2 * np.pi * 1j * xs)

    if flags & ECT_ANTIALIAS:
        kernel = antialias(kernel, xs, ys, 0.17, 0, np.log(R), np.log(R), 0.1)
        
    if flags & ECT_OFFSET_ORIGIN:
        shift = make_shift(image, offset, ect_offset, flags)

    image_padded = mod_image(image, ect_offset)
    out = xcorr(image_padded, kernel)
    out = out[:, :R][::-1, :]

    return shift * out if flags & ECT_OFFSET_ORIGIN else out